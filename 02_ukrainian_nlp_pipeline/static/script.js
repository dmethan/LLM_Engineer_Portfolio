const taskSelectElement = document.getElementById("taskSelect");
const inputTextElement = document.getElementById("inputText");
const contextTextElement = document.getElementById("contextText");
const submitBtnElement = document.getElementById("submitBtn");
const statusBadgeElement = document.getElementById("statusBadge");
const resultOutputElement = document.getElementById("resultOutput");

async function handleProcessClick() {
  const textValue = inputTextElement.value.trim();
  const contextValue = contextTextElement.value.trim();
  const taskValue = taskSelectElement.value;

  if (!textValue) {
    statusBadgeElement.textContent = "Помилка: введіть текст або запитання";
    return;
  }

  submitBtnElement.disabled = true;
  statusBadgeElement.textContent = "Обробка запиту моделлю...";

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: taskValue,
        text: textValue,
        context: contextValue,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      statusBadgeElement.textContent = `Помилка (${response.status})`;
      resultOutputElement.textContent = JSON.stringify(data, null, 2);
      return;
    }

    statusBadgeElement.textContent = `Виконано задачу: ${data.executed_task} (авто-роутинг: ${data.routed_automatically ? "так" : "ні"})`;
    resultOutputElement.textContent = JSON.stringify(data.result, null, 2);
  } catch (error) {
    statusBadgeElement.textContent = "Помилка з'єднання з сервером";
    resultOutputElement.textContent = String(error);
  } finally {
    submitBtnElement.disabled = false;
  }
}

submitBtnElement.addEventListener("click", handleProcessClick);