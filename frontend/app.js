import { createClient } from 'https://esm.sh/@supabase/supabase-js'

const API_BASE = "http://127.0.0.1:8000"
const SUPABASE_URL = "https://ucbbyzxhwjtkkwwefxah.supabase.co"
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjYmJ5enhod2p0a2t3d2VmeGFoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzcyMDQ4ODgsImV4cCI6MjA5Mjc4MDg4OH0.sL3rV-Q1bkUAWtjidSl1VlocQ9ADQhakQ64s8w8s2_A"

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

let accessToken = null
let currentSessionId = null
let sessions = []
const chunkCache = new Map()

const authView = document.getElementById("authView")
const chatView = document.getElementById("chatView")
const authForm = document.getElementById("authForm")
const signupBtn = document.getElementById("signupBtn")
const logoutBtn = document.getElementById("logoutBtn")
const authStatus = document.getElementById("authStatus")
const sessionList = document.getElementById("sessionList")
const chat = document.getElementById("chat")
const chatTitle = document.getElementById("chatTitle")
const messageForm = document.getElementById("messageForm")
const queryInput = document.getElementById("query")
const sendBtn = document.getElementById("sendBtn")
const newChatBtn = document.getElementById("newChatBtn")

function setAuthStatus(message, isError = false) {
  authStatus.textContent = message || ""
  authStatus.style.color = isError ? "#b42318" : ""
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${accessToken}`,
  }
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.headers || {}),
    },
  })

  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(data.detail || "Request failed")
  }
  return data
}

function showAuth() {
  authView.classList.remove("hidden")
  chatView.classList.add("hidden")
}

async function showChat() {
  authView.classList.add("hidden")
  chatView.classList.remove("hidden")
  renderEmptyChat()
  try {
    await loadSessions()
  } catch (error) {
    appendError(`Could not load chat history: ${error.message}`)
  }
}

function renderEmptyChat() {
  chatTitle.textContent = "New chat"
  chat.innerHTML = ""
  const empty = document.createElement("div")
  empty.className = "empty-state"
  empty.innerHTML = `
    <h2>Start a legal research chat</h2>
    <p>Ask about a section, offence, punishment, or legal concept. Your previous conversations stay available in the sidebar.</p>
  `
  chat.appendChild(empty)
}

function clearEmptyState() {
  const empty = chat.querySelector(".empty-state")
  if (empty) empty.remove()
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight
}

function textBlock(title, value) {
  if (!value) return null
  const section = document.createElement("section")
  section.className = "answer-section"

  const heading = document.createElement("h3")
  heading.textContent = title

  const body = document.createElement("p")
  body.textContent = String(value)

  section.append(heading, body)
  return section
}

function formatScore(result) {
  const raw = result?.confidence?.score ?? result?.confidence_score ?? result?.score
  if (raw === undefined || raw === null || raw === "") return "Not available"

  const numeric = Number(raw)
  if (Number.isFinite(numeric)) {
    return numeric <= 1 ? `${Math.round(numeric * 1000) / 10}%` : String(numeric)
  }
  return String(raw)
}

function riskClass(riskLevel) {
  const value = String(riskLevel || "").toLowerCase()
  if (value.includes("high")) return "risk-high"
  if (value.includes("medium") || value.includes("moderate")) return "risk-medium"
  if (value.includes("low")) return "risk-low"
  return ""
}

function metric(label, value, className = "") {
  const item = document.createElement("div")
  item.className = "metric"

  const name = document.createElement("span")
  name.textContent = label

  const val = document.createElement("strong")
  val.textContent = value || "Not available"
  if (className) val.classList.add(className)

  item.append(name, val)
  return item
}

function normalizedResult(result) {
  const data = result && typeof result === "object" ? result : {}
  return {
    detailedAnswer: data.detailed_answer || data.final_answer || data.answer || data.value || "",
    summaryAnswer: data.summary_answer || data.summary || "",
    score: formatScore(data),
    riskLevel: data.risk_level || "",
    riskReason: data.risk_reason || "",
    citations: Array.isArray(data.citations) ? data.citations : [],
    chunkIds: Array.isArray(data.selected_chunk_ids) ? data.selected_chunk_ids : [],
  }
}

function createAssistantContent(result) {
  const clean = normalizedResult(result)
  const wrapper = document.createElement("div")
  wrapper.className = "answer-grid"

  const detailed = textBlock("Detailed answer", clean.detailedAnswer)
  const summary = textBlock("Summary answer", clean.summaryAnswer)
  if (detailed) wrapper.appendChild(detailed)
  if (summary) wrapper.appendChild(summary)

  const meta = document.createElement("div")
  meta.className = "meta-grid"
  meta.append(
    metric("Score", clean.score),
    metric("Risk level", clean.riskLevel || "Not available", riskClass(clean.riskLevel))
  )
  wrapper.appendChild(meta)

  const riskReason = textBlock("Risk reason", clean.riskReason)
  if (riskReason) wrapper.appendChild(riskReason)

  if (clean.citations.length) {
    const citations = document.createElement("section")
    citations.className = "answer-section"

    const heading = document.createElement("h3")
    heading.textContent = "Citations"

    const list = document.createElement("div")
    list.className = "citation-list"

    clean.citations.forEach((citation, index) => {
      list.appendChild(createCitation(citation, clean.chunkIds[index]))
    })

    citations.append(heading, list)
    wrapper.appendChild(citations)
  }

  return wrapper
}

function createCitation(citationText, chunkId) {
  const item = document.createElement("article")
  item.className = "citation"

  const button = document.createElement("button")
  button.className = "citation-toggle"
  button.type = "button"
  button.textContent = citationText || "Citation"

  const body = document.createElement("div")
  body.className = "citation-body"
  body.textContent = "Loading citation text..."

  button.addEventListener("click", async () => {
    const isOpen = item.classList.toggle("open")
    if (!isOpen || item.dataset.loaded === "true") return

    if (!chunkId) {
      body.textContent = "No chunk id was returned for this citation."
      item.dataset.loaded = "true"
      return
    }

    try {
      const chunk = await loadChunk(chunkId)
      body.innerHTML = ""

      const meta = document.createElement("div")
      meta.className = "citation-meta"
      meta.textContent = [chunk.citation, chunk.section_title].filter(Boolean).join(" | ")

      const text = document.createElement("div")
      text.textContent = chunk.text || "No chunk text available."

      body.append(meta, text)
      item.dataset.loaded = "true"
    } catch (error) {
      body.textContent = error.message
    }
  })

  item.append(button, body)
  return item
}

async function loadChunk(chunkId) {
  if (chunkCache.has(chunkId)) {
    return chunkCache.get(chunkId)
  }
  const chunk = await apiFetch(`/chunks/${encodeURIComponent(chunkId)}`)
  chunkCache.set(chunkId, chunk)
  return chunk
}

function appendUserMessage(text) {
  clearEmptyState()
  const row = document.createElement("div")
  row.className = "message user"
  const bubble = document.createElement("div")
  bubble.className = "bubble"
  bubble.textContent = text
  row.appendChild(bubble)
  chat.appendChild(row)
  scrollToBottom()
}

function appendAssistantMessage(result) {
  clearEmptyState()
  const row = document.createElement("div")
  row.className = "message assistant"
  const bubble = document.createElement("div")
  bubble.className = "bubble"
  bubble.appendChild(createAssistantContent(result))
  row.appendChild(bubble)
  chat.appendChild(row)
  scrollToBottom()
}

function appendError(message) {
  clearEmptyState()
  const row = document.createElement("div")
  row.className = "message error"
  const bubble = document.createElement("div")
  bubble.className = "bubble"
  bubble.textContent = message
  row.appendChild(bubble)
  chat.appendChild(row)
  scrollToBottom()
}

function appendTyping() {
  clearEmptyState()
  const row = document.createElement("div")
  row.className = "message assistant"
  row.dataset.typing = "true"

  const bubble = document.createElement("div")
  bubble.className = "bubble"

  const typing = document.createElement("div")
  typing.className = "typing"
  typing.setAttribute("aria-label", "Assistant is thinking")
  typing.innerHTML = "<span></span><span></span><span></span>"

  bubble.appendChild(typing)
  row.appendChild(bubble)
  chat.appendChild(row)
  scrollToBottom()
  return row
}

function removeTyping(row) {
  if (row && row.parentElement) {
    row.remove()
  }
}

async function loadSessions() {
  const data = await apiFetch("/sessions")
  sessions = data.sessions || []
  renderSessions()
}

function renderSessions() {
  sessionList.innerHTML = ""

  if (!sessions.length) {
    const empty = document.createElement("div")
    empty.className = "session-item"
    empty.textContent = "No saved chats yet"
    sessionList.appendChild(empty)
    return
  }

  sessions.forEach((session) => {
    const button = document.createElement("button")
    button.className = "session-item"
    if (session.id === currentSessionId) button.classList.add("active")
    button.type = "button"
    button.textContent = session.title || "New chat"
    button.title = session.title || "New chat"
    button.addEventListener("click", () => loadHistory(session.id))
    sessionList.appendChild(button)
  })
}

async function loadHistory(sessionId) {
  currentSessionId = sessionId
  renderSessions()
  const session = sessions.find((item) => item.id === sessionId)
  chatTitle.textContent = session?.title || "Saved chat"
  chat.innerHTML = ""

  try {
    const data = await apiFetch(`/history/${encodeURIComponent(sessionId)}`)
    const messages = data.messages || []
    if (!messages.length) {
      renderEmptyChat()
      return
    }

    messages.forEach((message) => {
      if (message.role === "user") {
        appendUserMessage(message.content || "")
        return
      }

      if (message.role === "assistant") {
        appendAssistantMessage(parseAssistantContent(message.content))
      }
    })
  } catch (error) {
    appendError(error.message)
  }
}

function parseAssistantContent(content) {
  if (!content) return {}
  if (typeof content === "object") return content
  try {
    return JSON.parse(content)
  } catch {
    return { detailed_answer: String(content) }
  }
}

async function sendMessage() {
  const query = queryInput.value.trim()
  if (!query || sendBtn.disabled) return

  appendUserMessage(query)
  queryInput.value = ""
  resizeComposer()
  sendBtn.disabled = true
  const typing = appendTyping()

  try {
    const data = await apiFetch("/chat", {
      method: "POST",
      body: JSON.stringify({
        query,
        session_id: currentSessionId,
      }),
    })

    currentSessionId = data.session_id
    chatTitle.textContent = titleFromQuery(query)
    removeTyping(typing)
    appendAssistantMessage(data.result)
    loadSessions().catch((error) => appendError(`Could not refresh chat history: ${error.message}`))
  } catch (error) {
    removeTyping(typing)
    appendError(error.message)
  } finally {
    sendBtn.disabled = false
    queryInput.focus()
  }
}

function titleFromQuery(query) {
  return query.length > 60 ? `${query.slice(0, 57)}...` : query
}

function startNewChat() {
  currentSessionId = null
  renderSessions()
  renderEmptyChat()
  queryInput.value = ""
  queryInput.focus()
}

function resizeComposer() {
  queryInput.style.height = "auto"
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 160)}px`
}

async function login(email, password) {
  setAuthStatus("Signing in...")
  const { data, error } = await supabase.auth.signInWithPassword({ email, password })
  if (error) {
    setAuthStatus(error.message, true)
    return
  }

  accessToken = data.session.access_token
  setAuthStatus("")
  await showChat()
}

async function signup(email, password) {
  setAuthStatus("Creating account...")
  const { data, error } = await supabase.auth.signUp({ email, password })
  if (error) {
    setAuthStatus(error.message, true)
    return
  }

  if (data.session?.access_token) {
    accessToken = data.session.access_token
    setAuthStatus("")
    await showChat()
    return
  }

  setAuthStatus("Signup successful. Check your email if confirmation is enabled, then log in.")
}

async function logout() {
  await supabase.auth.signOut()
  accessToken = null
  currentSessionId = null
  sessions = []
  chunkCache.clear()
  showAuth()
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault()
  const email = document.getElementById("email").value.trim()
  const password = document.getElementById("password").value
  await login(email, password)
})

signupBtn.addEventListener("click", async () => {
  const email = document.getElementById("email").value.trim()
  const password = document.getElementById("password").value
  await signup(email, password)
})

logoutBtn.addEventListener("click", logout)
newChatBtn.addEventListener("click", startNewChat)
messageForm.addEventListener("submit", async (event) => {
  event.preventDefault()
  await sendMessage()
})

queryInput.addEventListener("input", resizeComposer)
queryInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    await sendMessage()
  }
})

supabase.auth.onAuthStateChange((_event, session) => {
  accessToken = session?.access_token || accessToken
})

const { data } = await supabase.auth.getSession()
if (data.session?.access_token) {
  accessToken = data.session.access_token
  await showChat()
} else {
  showAuth()
}
