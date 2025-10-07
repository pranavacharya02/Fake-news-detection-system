const apiBase = "http://127.0.0.1:5000";

async function checkHealth() {
  try {
    const res = await fetch(`${apiBase}/health`);
    const data = await res.json();
    return data;
  } catch (e) {
    return { status: "down" };
  }
}

document.getElementById("checkBtn").addEventListener("click", async () => {
  const text = document.getElementById("newsInput").value.trim();
  const resultDiv = document.getElementById("result");
  if (!text) {
    resultDiv.textContent = "⚠️ Please enter some text.";
    return;
  }
  resultDiv.textContent = "⏳ Checking...";

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (res.ok) {
      const tag = data.prediction === "FAKE" ? "❌ Fake News" : "✅ Real News";
      const conf = data.confidence ? ` (confidence ~ ${Math.round(data.confidence * 100)}%)` : "";
      resultDiv.textContent = `${tag}${conf}`;
    } else {
      resultDiv.textContent = `Error: ${data.error || "Unknown error"}`;
    }
  } catch (err) {
    const health = await checkHealth();
    if (health.status !== "ok") {
      resultDiv.textContent = "Server unreachable. Is the backend running on 127.0.0.1:5000?";
    } else {
      resultDiv.textContent = "An unexpected error occurred.";
    }
  }
});