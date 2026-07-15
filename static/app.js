const $ = (selector) => document.querySelector(selector);
const state = { history: [], busy: false, statusTimer: null };

const elements = {
  form: $("#chatForm"), input: $("#questionInput"), send: $("#sendButton"),
  messages: $("#messages"), hero: $("#hero"), scroll: $("#chatScroll"),
  newChat: $("#newChatButton"), reindex: $("#reindexButton"), category: $("#categorySelect"),
  documentCount: $("#documentCount"), chunkCount: $("#chunkCount"), categoryList: $("#categoryList"),
  statusDot: $("#statusDot"), statusLabel: $("#statusLabel"), modeHint: $("#modeHint"), toast: $("#toast")
};

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {"Content-Type": "application/json"}, ...options });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 3200);
}

function formatNumber(value) { return new Intl.NumberFormat().format(value || 0); }

function autosize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
}

function addMessage(role, content, sources = [], mode = "") {
  elements.hero.hidden = true;
  elements.messages.classList.add("active");
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const sourceMarkup = sources.length ? `
    <div class="sources"><div class="sources-title">Sources</div><div class="source-grid">
      ${sources.slice(0, 6).map((source, i) => `<a class="source-card" href="${escapeHtml(source.url)}" target="_blank" rel="noopener">
        <span class="source-number">[${i + 1}]</span><strong title="${escapeHtml(source.title)}">${escapeHtml(source.title)}</strong>
        <small>${escapeHtml(source.location)} · ${escapeHtml(source.category)}</small></a>`).join("")}
    </div></div>` : "";
  article.innerHTML = `<div class="avatar">${role === "user" ? "You" : "V"}</div>
    <div class="message-body"><div class="message-content">${escapeHtml(content)}</div>${sourceMarkup}
    ${mode ? `<span class="mode-badge">${mode === "llm" ? "AI synthesized" : "Evidence search"}</span>` : ""}</div>`;
  elements.messages.appendChild(article);
  elements.scroll.scrollTop = elements.scroll.scrollHeight;
  return article;
}

function addTyping() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = '<div class="avatar">V</div><div class="message-body"><div class="typing"><i></i><i></i><i></i></div></div>';
  elements.messages.appendChild(article);
  elements.scroll.scrollTop = elements.scroll.scrollHeight;
  return article;
}

async function ask(question) {
  if (!question || state.busy) return;
  state.busy = true;
  elements.send.disabled = true;
  addMessage("user", question);
  const typing = addTyping();
  elements.input.value = "";
  autosize();
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({question, history: state.history, category: elements.category.value})
    });
    typing.remove();
    addMessage("assistant", payload.answer, payload.sources, payload.mode);
    state.history.push({role: "user", content: question}, {role: "assistant", content: payload.answer});
    state.history = state.history.slice(-12);
  } catch (error) {
    typing.remove();
    addMessage("assistant", `I couldn't complete that search. ${error.message}`);
  } finally {
    state.busy = false;
    elements.send.disabled = false;
    elements.input.focus();
  }
}

async function refreshStats() {
  try {
    const stats = await api("/api/stats");
    elements.documentCount.textContent = formatNumber(stats.documents);
    elements.chunkCount.textContent = formatNumber(stats.chunks);
    const selected = elements.category.value;
    elements.category.innerHTML = '<option value="">All product categories</option>' + stats.categories.map(item =>
      `<option value="${escapeHtml(item.category)}">${escapeHtml(item.category)} (${item.count})</option>`).join("");
    elements.category.value = selected;
    elements.categoryList.innerHTML = stats.categories.slice(0, 5).map(item =>
      `<div class="category-row"><strong>${escapeHtml(item.category)}</strong><span>${item.count}</span></div>`).join("");
    elements.modeHint.textContent = stats.llm.enabled
      ? `Answers are synthesized with ${stats.llm.model} and grounded in cited documents.`
      : "ChromaDB RAG mode is active. Set LLM_MODEL for synthesized conversational answers.";
  } catch (error) { showToast(error.message); }
}

async function pollStatus() {
  try {
    const status = await api("/api/index/status");
    if (status.running) {
      const percent = status.total ? Math.round(status.processed / status.total * 100) : 0;
      elements.statusDot.className = "status-dot";
      elements.statusLabel.textContent = `Indexing ${status.processed}/${status.total} · ${percent}%`;
      elements.reindex.disabled = true;
    } else {
      elements.statusDot.className = "status-dot ready";
      elements.statusLabel.textContent = status.finished_at ? "Knowledge base ready" : "Ready to index";
      elements.reindex.disabled = false;
      await refreshStats();
    }
  } catch {
    elements.statusDot.className = "status-dot error";
    elements.statusLabel.textContent = "Server unavailable";
  }
  state.statusTimer = setTimeout(pollStatus, 1800);
}

elements.form.addEventListener("submit", event => { event.preventDefault(); ask(elements.input.value.trim()); });
elements.input.addEventListener("input", autosize);
elements.input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); }
});
document.addEventListener("click", event => {
  const suggestion = event.target.closest("[data-question]");
  if (suggestion) ask(suggestion.dataset.question);
});
elements.newChat.addEventListener("click", () => {
  state.history = [];
  elements.messages.innerHTML = "";
  elements.messages.classList.remove("active");
  elements.hero.hidden = false;
  elements.input.focus();
});
elements.reindex.addEventListener("click", async () => {
  try {
    await api("/api/index/start", {method: "POST", body: JSON.stringify({force: false})});
    showToast("Knowledge-base refresh started");
    pollStatus();
  } catch (error) { showToast(error.message); }
});

refreshStats();
pollStatus();
elements.input.focus();
