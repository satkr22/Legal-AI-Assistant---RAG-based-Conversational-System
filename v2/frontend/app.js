import { createClient } from 'https://esm.sh/@supabase/supabase-js'

const config = window.LEGAL_AI_CONFIG || {}
const configuredApiBase = config.API_BASE && !config.API_BASE.includes("REPLACE_WITH_RAILWAY_DOMAIN")
  ? config.API_BASE
  : ""


// const API_BASE = (
//   configuredApiBase ||
//   (["localhost", "127.0.0.1", "0.0.0.1"].includes(window.location.hostname)
//     ? "http://127.0.0.1:8000"
//     : "")
// ).replace(/\/+$/, "")

// backend url here
const API_BASE = "https://air-mixer-jews-clone.trycloudflare.com"

const SUPABASE_URL = config.SUPABASE_URL
const SUPABASE_ANON_KEY = config.SUPABASE_ANON_KEY

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

let accessToken = null
let currentUser = null
let currentSessionId = null
let sessions = []
const chunkCache = new Map()
const ACTIVE_SESSION_STORAGE_KEY = "legal-ai-active-session"
const QUERY_DRAFT_STORAGE_KEY = "legal-ai-query-draft"
const THINKING_PHASES = [
  ["files", "Loading legal data"],
  ["direct_lookup", "Finding section text"],
  ["intent", "Assessing query"],
  ["retrieval", "Searching legal database"],
  ["rerank", "Reranking sources"],
  ["reasoning", "Building context"],
  ["guardrails", "Checking grounding"],
  ["validation", "Finalizing answer"],
]
const THINKING_PHASE_COPY = {
  files: {
    running: "Loading legal data",
    completed: "Legal data ready",
  },
  direct_lookup: {
    running: "Finding the exact section in the index",
    completed: "Section text resolved",
  },
  intent: {
    running: "Assessing query",
    completed: "Query assessed",
  },
  retrieval: {
    running: "Searching legal database",
    completed: "Sources found",
  },
  rerank: {
    running: "Reranking sources",
    completed: "Sources reranked",
  },
  reasoning: {
    running: "Building context from selected sources",
    completed: "Context prepared",
  },
  guardrails: {
    running: "Checking grounding and citations",
    completed: "Grounding checks complete",
  },
  validation: {
    running: "Finalizing answer",
    completed: "Answer ready",
  },
}

const authView = document.getElementById("authView")
const chatView = document.getElementById("chatView")
const authForm = document.getElementById("authForm")
const signupBtn = document.getElementById("signupBtn")
const googleLoginBtn = document.getElementById("googleLoginBtn")
const logoutBtn = document.getElementById("logoutBtn")
const authStatus = document.getElementById("authStatus")
const sessionList = document.getElementById("sessionList")
const chat = document.getElementById("chat")
const sidebarScroll = document.querySelector(".sidebar-scroll")
const messageForm = document.getElementById("messageForm")
const queryInput = document.getElementById("query")
const sendBtn = document.getElementById("sendBtn")
const newChatBtn = document.getElementById("newChatBtn")
const accountBtn = document.getElementById("accountBtn")
const accountPanel = document.getElementById("accountPanel")
const accountInitial = document.getElementById("accountInitial")
const accountEmailShort = document.getElementById("accountEmailShort")
const accountEmail = document.getElementById("accountEmail")
const accountProvider = document.getElementById("accountProvider")

function setAuthStatus(message, isError = false) {
  authStatus.textContent = message || ""
  authStatus.style.color = isError ? "var(--danger)" : ""
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${accessToken}`,
  }
}

function setSession(session) {
  accessToken = session?.access_token || null
  currentUser = session?.user || null
  updateAccountDetails()
}

function updateAccountDetails() {
  const email = currentUser?.email || "Not available"
  const provider = currentUser?.app_metadata?.provider || currentUser?.identities?.[0]?.provider || "email"
  const initial = email && email !== "Not available" ? email[0].toUpperCase() : "A"

  accountInitial.textContent = initial
  accountEmailShort.textContent = email
  accountEmail.textContent = email
  accountProvider.textContent = provider.charAt(0).toUpperCase() + provider.slice(1)
}

function userStorageKey(key) {
  return `${key}:${currentUser?.id || "anonymous"}`
}

function getStoredValue(key) {
  try {
    return window.localStorage.getItem(userStorageKey(key))
  } catch {
    return null
  }
}

function setStoredValue(key, value) {
  try {
    if (value) {
      window.localStorage.setItem(userStorageKey(key), value)
      return
    }
    window.localStorage.removeItem(userStorageKey(key))
  } catch {
    // Storage can be unavailable in private browsing or restricted contexts.
  }
}

function rememberCurrentSession() {
  setStoredValue(ACTIVE_SESSION_STORAGE_KEY, currentSessionId)
}

function rememberQueryDraft() {
  setStoredValue(QUERY_DRAFT_STORAGE_KEY, queryInput.value)
}

function restoreQueryDraft() {
  queryInput.value = getStoredValue(QUERY_DRAFT_STORAGE_KEY) || ""
  resizeComposer()
}

function closeAccountPanel() {
  accountPanel.classList.add("hidden")
  accountBtn.setAttribute("aria-expanded", "false")
}

async function apiFetch(path, options = {}) {
  if (!API_BASE || API_BASE.includes("REPLACE_WITH_RAILWAY_DOMAIN")) {
    throw new Error("Backend API URL is not configured. Update frontend/config.js with your Railway public URL.")
  }

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

async function apiStream(path, body, onEvent) {
  if (!API_BASE || API_BASE.includes("REPLACE_WITH_RAILWAY_DOMAIN")) {
    throw new Error("Backend API URL is not configured. Update frontend/config.js with your Railway public URL.")
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Accept": "text/event-stream",
    },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || "Request failed")
  }
  if (!res.body) {
    throw new Error("Streaming is not supported by this browser.")
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split("\n\n")
    buffer = events.pop() || ""
    events.forEach((rawEvent) => {
      const parsed = parseSseEvent(rawEvent)
      if (parsed) onEvent(parsed)
    })
  }

  if (buffer.trim()) {
    const parsed = parseSseEvent(buffer)
    if (parsed) onEvent(parsed)
  }
}

function parseSseEvent(rawEvent) {
  const lines = rawEvent.split("\n")
  let event = "message"
  let data = ""
  lines.forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim()
    if (line.startsWith("data:")) data += line.slice(5).trim()
  })
  if (!data) return null
  try {
    return { event, data: JSON.parse(data) }
  } catch {
    return { event, data: { value: data } }
  }
}

function showAuth() {
  authView.classList.remove("hidden")
  chatView.classList.add("hidden")
}

async function showChat() {
  authView.classList.add("hidden")
  chatView.classList.remove("hidden")
  closeAccountPanel()
  renderEmptyChat()
  try {
    await loadSessions()
    restoreQueryDraft()
    const storedSessionId = getStoredValue(ACTIVE_SESSION_STORAGE_KEY)
    const hasStoredSession = storedSessionId && sessions.some((session) => session.id === storedSessionId)
    if (hasStoredSession) {
      await loadHistory(storedSessionId)
    } else {
      currentSessionId = null
      rememberCurrentSession()
      renderSessions()
    }
  } catch (error) {
    appendError(`Could not load chat history: ${error.message}`)
  }
}

function renderEmptyChat() {
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

function formatScoreValue(raw) {
  if (raw === undefined || raw === null || raw === "") return "Not available"

  const numeric = Number(raw)
  if (Number.isFinite(numeric)) {
    return numeric <= 1 ? `${Math.round(numeric * 1000) / 10}%` : String(numeric)
  }
  return String(raw)
}

function formatAnswerScore(result) {
  return formatScoreValue(result?.confidence?.score ?? result?.confidence_score ?? result?.score)
}

function formatRetrievalScore(result) {
  return formatScoreValue(result?.retrieval_confidence?.score)
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
    answerType: data.answer_type || "",
    sectionNumber: data.section_number || "",
    sectionTitle: data.section_title || "",
    sectionCitation: data.citation || "",
    sectionText: data.text || "",
    parts: Array.isArray(data.parts) ? data.parts : [],
    detailedAnswer: data.detailed_answer || data.final_answer || data.answer || data.value || "",
    summaryAnswer: data.summary_answer || data.summary || "",
    answerScore: formatAnswerScore(data),
    retrievalScore: formatRetrievalScore(data),
    riskLevel: data.risk_level || "",
    riskReason: data.risk_reason || "",
    citations: Array.isArray(data.citations) ? data.citations : [],
    chunkIds: Array.isArray(data.selected_chunk_ids) ? data.selected_chunk_ids : [],
  }
}

function createAssistantContent(result) {
  const clean = normalizedResult(result)
  if (clean.answerType === "direct_section_lookup") {
    return createDirectSectionContent(clean)
  }

  const wrapper = document.createElement("div")
  wrapper.className = "answer-grid"

  const detailed = textBlock("Detailed answer", clean.detailedAnswer)
  const summary = textBlock("Summary answer", clean.summaryAnswer)
  if (detailed) wrapper.appendChild(detailed)
  if (summary) wrapper.appendChild(summary)

  const meta = document.createElement("div")
  meta.className = "meta-grid"
  meta.append(
    metric("Answer confidence", clean.answerScore),
    metric("Retrieval confidence", clean.retrievalScore),
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

function createDirectSectionContent(clean) {
  const wrapper = document.createElement("div")
  wrapper.className = "answer-grid direct-section"

  const section = document.createElement("section")
  section.className = "answer-section"

  const heading = document.createElement("h3")
  heading.textContent = `Section ${clean.sectionNumber}${clean.sectionTitle ? `: ${clean.sectionTitle}` : ""}`

  const meta = document.createElement("div")
  meta.className = "citation-meta"
  meta.textContent = clean.sectionCitation || "Bharatiya Nyaya Sanhita, 2023"

  const summary = document.createElement("p")
  summary.textContent = clean.summaryAnswer

  const scores = document.createElement("div")
  scores.className = "meta-grid section-confidence"
  scores.append(
    metric("Answer confidence", clean.answerScore),
    metric("Retrieval confidence", clean.retrievalScore),
    metric("Risk level", clean.riskLevel || "Not available", riskClass(clean.riskLevel))
  )

  section.append(heading, meta, summary, scores)
  wrapper.appendChild(section)

  if (clean.parts.length) {
    const partsSection = document.createElement("section")
    partsSection.className = "answer-section"

    const partsHeading = document.createElement("h3")
    partsHeading.textContent = "Parts"

    const list = document.createElement("div")
    list.className = "section-parts"
    clean.parts.forEach((part, index) => {
      list.appendChild(createSectionPart(part, index))
    })

    partsSection.append(partsHeading, list)
    wrapper.appendChild(partsSection)
  }

  return wrapper
}

function createSectionPart(part, index) {
  const item = document.createElement("article")
  item.className = "section-part"

  const button = document.createElement("button")
  button.className = "section-part-toggle"
  button.type = "button"
  button.textContent = part.citation || part.label || `Part ${index + 1}`

  const body = document.createElement("div")
  body.className = "section-part-body"

  const meta = document.createElement("div")
  meta.className = "citation-meta"
  meta.textContent = [part.chunk_type, part.label].filter(Boolean).join(" | ")

  const text = document.createElement("div")
  text.textContent = part.text || "No text available."

  body.append(meta, text)

  button.addEventListener("click", () => {
    item.classList.toggle("open")
  })

  item.append(button, body)
  return item
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

function appendThinkingPanel() {
  clearEmptyState()
  const row = document.createElement("div")
  row.className = "message assistant thinking-message"
  row.dataset.thinking = "true"

  const bubble = document.createElement("div")
  bubble.className = "bubble thinking-bubble"

  const panel = createThinkingPanel()
  bubble.appendChild(panel.root)
  row.appendChild(bubble)
  chat.appendChild(row)
  scrollToBottom()
  return { row, panel }
}

function createThinkingPanel() {
  const root = document.createElement("section")
  root.className = "thinking-panel"

  const header = document.createElement("button")
  header.className = "thinking-header"
  header.type = "button"

  const left = document.createElement("span")
  left.className = "thinking-header-left"

  const mark = document.createElement("span")
  mark.className = "thinking-mark"
  mark.textContent = "✣"

  const title = document.createElement("span")
  title.textContent = "Show reasoning"

  const status = document.createElement("strong")
  status.textContent = "Starting"

  const chevron = document.createElement("span")
  chevron.className = "thinking-chevron"
  chevron.textContent = "⌄"

  left.append(mark, title)
  header.append(left, status, chevron)

  const body = document.createElement("div")
  body.className = "thinking-body"

  const phaseList = document.createElement("div")
  phaseList.className = "thinking-phases"
  const phaseEls = new Map()
  THINKING_PHASES.forEach(([id, label]) => {
    const item = document.createElement("div")
    item.className = "thinking-phase pending"
    item.dataset.phase = id

    const icon = document.createElement("span")
    icon.className = "thinking-phase-icon"
    icon.textContent = "○"

    const text = document.createElement("div")
    const name = document.createElement("strong")
    name.textContent = label
    const detail = document.createElement("p")
    detail.textContent = ""

    text.append(name, detail)
    item.append(icon, text)
    phaseList.appendChild(item)
    phaseEls.set(id, { item, icon, name, detail })
  })

  const evidence = createThinkingSection("Evidence details")
  evidence.root.classList.add("thinking-evidence-section")

  body.append(phaseList, evidence.root)
  root.append(header, body)

  header.addEventListener("click", () => {
    root.classList.toggle("collapsed")
  })

  return {
    root,
    status,
    phaseEls,
    sections: { evidence },
    retrievalCount: 0,
    rerankCount: 0,
    selectedChunkIds: [],
    selectedChunks: [],
  }
}

function createThinkingSection(title) {
  const root = document.createElement("div")
  root.className = "thinking-section"
  const heading = document.createElement("h3")
  heading.textContent = title
  const content = document.createElement("div")
  content.className = "thinking-section-content"
  root.append(heading, content)
  return { root, content }
}

function thinkingPhaseText(id, status, fallback) {
  const copy = THINKING_PHASE_COPY[id]
  if (copy && copy[status]) return copy[status]
  return fallback || "Working"
}

function chunkIdList(chunks = [], limit = 4) {
  return chunks
    .map((chunk) => chunk.chunk_id)
    .filter(Boolean)
    .slice(0, limit)
    .join(", ")
}

function thinkingPhaseDetail(id, data = {}) {
  if (id === "files") return "Opening the indexed BNS chunk corpus."
  if (id === "direct_lookup") return data.detail || "Resolving the exact section and its child nodes."
  if (id === "intent" && data.data) {
    const intent = data.data.intent?.primary
    const concepts = (data.data.concepts || []).slice(0, 3).join(", ")
    return [intent ? `Intent: ${intent}` : "", concepts ? `Concepts: ${concepts}` : ""].filter(Boolean).join(" · ")
  }
  if (id === "retrieval" && data.status === "running") return "Scanning candidate sections and legal concepts."
  if (id === "rerank" && data.status === "running") return "Comparing candidate chunks for relevance."
  if (id === "reasoning" && data.status === "running") return "Preparing the final prompt context from selected evidence."
  if (id === "guardrails") return "Checking citations, grounding, and unsupported claims."
  if (id === "validation") return "Preparing the final response for display."
  return data.detail || ""
}

function updateThinkingPhase(panel, data) {
  const phase = panel.phaseEls.get(data.id)
  if (!phase) return
  phase.item.classList.remove("pending", "running", "completed", "failed")
  phase.item.classList.add(data.status || "running")
  phase.icon.textContent = data.status === "completed" ? "✓" : data.status === "failed" ? "!" : "○"
  const text = thinkingPhaseText(data.id, data.status || "running", data.label)
  phase.name.textContent = text
  const detail = thinkingPhaseDetail(data.id, data)
  const keepEvidenceDetail = data.status === "completed"
    && ["retrieval", "rerank", "reasoning"].includes(data.id)
    && phase.detail.textContent
  if (!keepEvidenceDetail) {
    phase.detail.textContent = detail
    phase.detail.hidden = !detail
  }
  panel.status.textContent = text
  scrollToBottom()
}

function completeThinkingPanel(panel) {
  panel.status.textContent = "Reasoning complete"
  panel.root.classList.add("complete", "collapsed")
  panel.phaseEls.forEach((phase) => {
    if (phase.item.classList.contains("running")) {
      phase.item.classList.remove("running")
      phase.item.classList.add("completed")
      phase.icon.textContent = "✓"
    }
  })
  const validation = panel.phaseEls.get("validation")
  if (validation && !validation.item.classList.contains("completed")) {
    validation.item.classList.remove("pending", "running", "failed")
    validation.item.classList.add("completed")
    validation.icon.textContent = "✓"
    validation.name.textContent = "Answer ready"
    validation.detail.textContent = "Final response prepared."
    validation.detail.hidden = false
  }
}

function renderChunkCards(chunks = []) {
  const list = document.createElement("div")
  list.className = "thinking-chunks"
  chunks.forEach((chunk) => {
    const card = document.createElement("article")
    card.className = "thinking-chunk"
    if (chunk.selected) card.classList.add("selected")

    const title = document.createElement("strong")
    title.textContent = `#${chunk.rank || "-"} ${chunk.chunk_id || "chunk"}`

    const meta = document.createElement("div")
    meta.className = "thinking-chunk-meta"
    meta.textContent = [
      chunk.citation ? `Citation: ${chunk.citation}` : "",
      chunk.retrieval_score !== null && chunk.retrieval_score !== undefined ? `Retrieval: ${chunk.retrieval_score}` : "",
      chunk.rerank_score !== null && chunk.rerank_score !== undefined ? `Re-rank: ${chunk.rerank_score}` : "",
    ].filter(Boolean).join(" | ")

    const preview = document.createElement("p")
    preview.textContent = chunk.preview || "No preview available."

    card.append(title, meta, preview)
    list.appendChild(card)
  })
  return list
}

function renderEvidenceDetails(panel) {
  const content = panel.sections.evidence.content
  content.innerHTML = ""

  const selectedCount = panel.selectedChunkIds.length
  const summary = document.createElement("button")
  summary.className = "thinking-evidence-toggle"
  summary.type = "button"
  summary.textContent = [
    panel.retrievalCount ? `${panel.retrievalCount} retrieved` : "",
    panel.rerankCount ? `${panel.rerankCount} reranked` : "",
    selectedCount ? `${selectedCount} selected` : "",
  ].filter(Boolean).join(" · ") || "Waiting for evidence"

  const details = document.createElement("div")
  details.className = "thinking-evidence-details"
  if (panel.selectedChunks.length) {
    details.appendChild(renderChunkCards(panel.selectedChunks.slice(0, 5)))
  } else if (panel.selectedChunkIds.length) {
    const ids = document.createElement("div")
    ids.className = "selected-chip-list"
    panel.selectedChunkIds.forEach((id) => {
      const chip = document.createElement("span")
      chip.className = "selected-chip"
      chip.textContent = id
      ids.appendChild(chip)
    })
    details.appendChild(ids)
  }

  summary.addEventListener("click", () => {
    panel.sections.evidence.root.classList.toggle("open")
  })

  content.append(summary, details)
}

function updateThinkingChunks(panel, data) {
  const chunks = data.chunks || []
  if (data.stage === "reranked") {
    panel.rerankCount = chunks.length
    const phase = panel.phaseEls.get("rerank")
    const leaders = chunkIdList(chunks, 4)
    if (phase && leaders) {
      phase.detail.textContent = `Leading chunks: ${leaders}`
      phase.detail.hidden = false
    }
  } else {
    panel.retrievalCount = chunks.length
    const phase = panel.phaseEls.get("retrieval")
    const found = chunkIdList(chunks, 4)
    if (phase && found) {
      phase.detail.textContent = `Found chunks: ${found}`
      phase.detail.hidden = false
    }
  }
  renderEvidenceDetails(panel)
}

function updateThinkingSelection(panel, data) {
  const ids = new Set(data.selected_chunk_ids || [])
  panel.selectedChunkIds = [...ids]
  panel.selectedChunks = (data.chunks || []).filter((chunk) => ids.has(chunk.chunk_id))
  const phase = panel.phaseEls.get("reasoning")
  if (phase && panel.selectedChunkIds.length) {
    phase.detail.textContent = `Using context: ${panel.selectedChunkIds.slice(0, 5).join(", ")}`
    phase.detail.hidden = false
  }
  renderEvidenceDetails(panel)
}

function handleThinkingEvent(panel, parsed) {
  const { event, data } = parsed
  if (event === "phase") updateThinkingPhase(panel, data)
  if (event === "chunks") updateThinkingChunks(panel, data)
  if (event === "selection") updateThinkingSelection(panel, data)
  if (event === "error") {
    panel.status.textContent = "Error"
    updateThinkingPhase(panel, {
      id: "reasoning",
      label: "Stream failed",
      status: "failed",
      progress: 1,
      detail: data.message || "The stream failed.",
    })
  }
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
  rememberCurrentSession()
  renderSessions()
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
  rememberQueryDraft()
  resizeComposer()
  sendBtn.disabled = true
  const thinking = appendThinkingPanel()
  updateThinkingPhase(thinking.panel, {
    id: "files",
    label: "Opening live stream",
    status: "running",
    progress: 0.02,
    detail: "Connecting to the backend progress stream.",
  })

  try {
    let finalResult = null
    await apiStream(
      "/chat/stream",
      {
        query,
        session_id: currentSessionId,
      },
      ({ event, data }) => {
        if (event === "session" && data.session_id) {
          currentSessionId = data.session_id
          rememberCurrentSession()
          renderSessions()
          return
        }
        if (event === "final") {
          currentSessionId = data.session_id || currentSessionId
          rememberCurrentSession()
          finalResult = data.result
          return
        }
        handleThinkingEvent(thinking.panel, { event, data })
      }
    )

    if (finalResult) {
      completeThinkingPanel(thinking.panel)
      appendAssistantMessage(finalResult)
    }
    loadSessions().catch((error) => appendError(`Could not refresh chat history: ${error.message}`))
  } catch (error) {
    handleThinkingEvent(thinking.panel, { event: "error", data: { message: error.message } })
    appendError(error.message)
  } finally {
    sendBtn.disabled = false
    queryInput.focus()
  }
}

function startNewChat() {
  currentSessionId = null
  rememberCurrentSession()
  renderSessions()
  renderEmptyChat()
  queryInput.value = ""
  rememberQueryDraft()
  resizeComposer()
  queryInput.focus()
}

function resizeComposer() {
  queryInput.style.height = "auto"
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 150)}px`
}

function revealScrollbarWhileScrolling(element) {
  if (!element) return

  let timeoutId
  element.addEventListener("scroll", () => {
    element.classList.add("is-scrolling")
    window.clearTimeout(timeoutId)
    timeoutId = window.setTimeout(() => {
      element.classList.remove("is-scrolling")
    }, 900)
  }, { passive: true })
}

async function login(email, password) {
  setAuthStatus("Signing in...")
  const { data, error } = await supabase.auth.signInWithPassword({ email, password })
  if (error) {
    setAuthStatus(error.message, true)
    return
  }

  setSession(data.session)
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
    setSession(data.session)
    setAuthStatus("")
    await showChat()
    return
  }

  setAuthStatus("Signup successful. Check your email if confirmation is enabled, then log in.")
}

async function loginWithGoogle() {
  setAuthStatus("Opening Google sign in...")
  const redirectTo = `${window.location.origin}${window.location.pathname}`
  const { error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo,
    },
  })

  if (error) {
    setAuthStatus(error.message, true)
  }
}

async function logout() {
  await supabase.auth.signOut()
  setSession(null)
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

googleLoginBtn.addEventListener("click", loginWithGoogle)
logoutBtn.addEventListener("click", logout)
newChatBtn.addEventListener("click", startNewChat)
accountBtn.addEventListener("click", () => {
  const isOpening = accountPanel.classList.contains("hidden")
  accountPanel.classList.toggle("hidden", !isOpening)
  accountBtn.setAttribute("aria-expanded", String(isOpening))
})

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault()
  await sendMessage()
})

queryInput.addEventListener("input", resizeComposer)
queryInput.addEventListener("input", rememberQueryDraft)
queryInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    await sendMessage()
  }
})

revealScrollbarWhileScrolling(chat)
revealScrollbarWhileScrolling(sidebarScroll)
revealScrollbarWhileScrolling(queryInput)

supabase.auth.onAuthStateChange(async (event, session) => {
  setSession(session)
  if (event === "SIGNED_IN" && chatView.classList.contains("hidden")) {
    setAuthStatus("")
    await showChat()
    return
  }

  if (event === "SIGNED_OUT") {
    showAuth()
  }
})

const { data } = await supabase.auth.getSession()
if (data.session?.access_token) {
  setSession(data.session)
  await showChat()
} else {
  setSession(null)
  showAuth()
}
