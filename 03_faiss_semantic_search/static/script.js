const searchQueryEl = document.getElementById("searchQuery");
const topKInputEl = document.getElementById("topKInput");
const searchBtnEl = document.getElementById("searchBtn");
const searchStatusEl = document.getElementById("searchStatus");
const tokenInfoBoxEl = document.getElementById("tokenInfoBox");
const tokenDetailsEl = document.getElementById("tokenDetails");
const searchResultsEl = document.getElementById("searchResults");

const maskSentenceEl = document.getElementById("maskSentence");
const predictBtnEl = document.getElementById("predictBtn");
const predictStatusEl = document.getElementById("predictStatus");
const predictResultsEl = document.getElementById("predictResults");

async function runSemanticSearch() {
  const query = searchQueryEl.value.trim();
  const topK = Number(topKInputEl.value) || 5;

  if (!query) {
    searchStatusEl.textContent = "Введіть пошуковий запит.";
    return;
  }

  searchBtnEl.disabled = true;
  searchStatusEl.textContent = "Виконується токенізація, побудова ембедингу та пошук у FAISS...";
  searchResultsEl.innerHTML = "";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    const data = await response.json();

    if (!response.ok) {
      searchStatusEl.textContent = `Помилка (${response.status}): ${data.detail || "невідома помилка"}`;
      return;
    }

    searchStatusEl.textContent = `Модель: ${data.model_name} | Розмірність ембедингу: ${data.embedding_dim} | Знайдено: ${data.results.length}`;

    tokenInfoBoxEl.classList.remove("hidden");
    const tokensPreview = data.tokenization.tokens.join(" | ");
    const idsPreview = data.tokenization.input_ids.join(", ");
    tokenDetailsEl.innerHTML = `
      <div><strong>Кількість токенів:</strong> ${data.tokenization.tokens_count}</div>
      <div><strong>Токени:</strong> ${tokensPreview}</div>
      <div><strong>input_ids:</strong> [${idsPreview}]</div>
    `;

    data.results.forEach((item) => {
      const card = document.createElement("div");
      card.className = "result-item";
      card.innerHTML = `
        <div class="result-meta">#${item.rank} · Категорія: ${item.category} · Схожість (cosine): ${item.score} · doc_id: ${item.doc_id}</div>
        <p class="result-text">${item.text}</p>
      `;
      searchResultsEl.appendChild(card);
    });
  } catch (err) {
    searchStatusEl.textContent = `Помилка з'єднання: ${String(err)}`;
  } finally {
    searchBtnEl.disabled = false;
  }
}

async function runTokenPrediction() {
  const sentence = maskSentenceEl.value.trim();

  if (!sentence) {
    predictStatusEl.textContent = "Введіть речення для передбачення токенів.";
    return;
  }

  predictBtnEl.disabled = true;
  predictStatusEl.textContent = "Модель обчислює ймовірності токенів...";
  predictResultsEl.innerHTML = "";

  try {
    const response = await fetch("/api/predict-tokens", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence, top_n: 5 }),
    });
    const data = await response.json();

    if (!response.ok) {
      predictStatusEl.textContent = `Помилка (${response.status}): ${data.detail || "невідома помилка"}`;
      return;
    }

    predictStatusEl.textContent = `Модель: ${data.model_name} | Речення: "${data.sentence}"`;

    data.predictions.forEach((pred, idx) => {
      const card = document.createElement("div");
      card.className = "result-item";
      card.innerHTML = `
        <div class="result-meta">Варіант #${idx + 1} · Токен: "${pred.token}" · Ймовірність: ${pred.probability}</div>
        <p class="result-text">${pred.completed_text}</p>
      `;
      predictResultsEl.appendChild(card);
    });
  } catch (err) {
    predictStatusEl.textContent = `Помилка з'єднання: ${String(err)}`;
  } finally {
    predictBtnEl.disabled = false;
  }
}

searchBtnEl.addEventListener("click", runSemanticSearch);
predictBtnEl.addEventListener("click", runTokenPrediction);