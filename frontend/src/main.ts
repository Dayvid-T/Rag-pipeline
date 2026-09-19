import {
  askQuestion,
  deleteDocument,
  formatLatency,
  listDocuments,
  uploadDocument,
  type DocumentInfo,
} from "./api";
import { forgetCitation, openCitation } from "./citation";

const form = document.getElementById("ask-form") as HTMLFormElement;
const questionInput = document.getElementById("question") as HTMLTextAreaElement;
const askButton = document.getElementById("ask-button") as HTMLButtonElement;
const emptyEl = document.getElementById("empty") as HTMLElement;
const result = document.getElementById("result") as HTMLElement;
const guardrailNotice = document.getElementById("guardrail-notice") as HTMLElement;
const guardrailTitle = document.getElementById("guardrail-title") as HTMLElement;
const guardrailDetail = document.getElementById("guardrail-detail") as HTMLElement;
const answerEl = document.getElementById("answer") as HTMLElement;
const sourcesEl = document.getElementById("sources") as HTMLUListElement;
const contextsEl = document.getElementById("contexts") as HTMLOListElement;
const latencyEl = document.getElementById("latency") as HTMLElement;
const errorEl = document.getElementById("error") as HTMLElement;

const dropzone = document.getElementById("dropzone") as HTMLElement;
const fileInput = document.getElementById("file") as HTMLInputElement;
const dropOverlay = document.getElementById("drop-overlay") as HTMLElement;
const documentsEl = document.getElementById("documents") as HTMLUListElement;
const documentsCount = document.getElementById("documents-count") as HTMLElement;
const documentsStatus = document.getElementById("documents-status") as HTMLElement;

// --- Ask ------------------------------------------------------------------

function setBusy(busy: boolean) {
  askButton.disabled = busy;
  questionInput.disabled = busy;
  askButton.textContent = busy ? "Thinking…" : "Ask";
}

function renderList(container: HTMLElement, items: string[]) {
  container.replaceChildren(
    ...items.map((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      return li;
    }),
  );
}

function renderGuardrailNotice(blocked: boolean, flags: string[]) {
  if (blocked) {
    guardrailNotice.classList.remove("is-filtered");
    guardrailNotice.classList.add("is-blocked");
    guardrailTitle.textContent = "Blocked by a guardrail";
    guardrailDetail.textContent = flags.length
      ? `Flagged: ${flags.join(", ")}`
      : "This request was refused before an answer was generated.";
    guardrailNotice.hidden = false;
  } else if (flags.length > 0) {
    guardrailNotice.classList.remove("is-blocked");
    guardrailNotice.classList.add("is-filtered");
    guardrailTitle.textContent = "A retrieved passage was filtered";
    guardrailDetail.textContent = `Excluded from the answer for: ${flags.join(", ")}`;
    guardrailNotice.hidden = false;
  } else {
    guardrailNotice.hidden = true;
  }
}

async function submit() {
  const question = questionInput.value.trim();
  if (!question) return;

  errorEl.hidden = true;
  setBusy(true);
  try {
    const data = await askQuestion(question);
    answerEl.textContent = data.answer;
    renderList(sourcesEl, data.sources);
    renderList(contextsEl, data.contexts);
    renderGuardrailNotice(data.blocked, data.guardrail_flags);
    latencyEl.textContent = formatLatency(data.latencyMs);
    emptyEl.hidden = true;
    result.hidden = false;
  } catch (err) {
    result.hidden = true;
    emptyEl.hidden = true;
    errorEl.textContent = err instanceof Error ? err.message : "Something went wrong.";
    errorEl.hidden = false;
  } finally {
    setBusy(false);
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void submit();
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void submit();
  }
});

// --- Library ----------------------------------------------------------------

type StatusKind = "info" | "error" | "success";

function setStatus(message: string, kind: StatusKind = "info") {
  documentsStatus.textContent = message;
  documentsStatus.classList.toggle("status-error", kind === "error");
  documentsStatus.classList.toggle("status-success", kind === "success");
}

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function documentRow(doc: DocumentInfo, pending = false): HTMLLIElement {
  const li = document.createElement("li");
  li.dataset.source = doc.source;
  if (pending) li.classList.add("is-pending");

  const icon = document.createElement("span");
  icon.className = "doc-icon";
  icon.textContent = doc.source.toLowerCase().endsWith(".pdf") ? "PDF" : "TXT";

  const body = document.createElement("div");
  body.className = "doc-body";
  const name = document.createElement("div");
  name.className = "doc-name";
  name.textContent = doc.source;
  name.title = doc.source;
  const meta = document.createElement("div");
  meta.className = "doc-meta";
  meta.textContent = pending ? "Indexing…" : plural(doc.chunks, "chunk");
  body.append(name, meta);

  const cite = document.createElement("button");
  cite.type = "button";
  cite.className = "ghost-button";
  cite.textContent = "Cite";
  cite.title = "Reference in IEEE / APA";
  cite.disabled = pending;
  cite.addEventListener("click", () => void openCitation(doc.source));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "icon-button";
  remove.setAttribute("aria-label", `Remove ${doc.source}`);
  remove.title = "Remove";
  remove.textContent = "×";
  remove.disabled = pending;
  remove.addEventListener("click", () => void removeDocument(doc.source, remove));

  li.append(icon, body, cite, remove);
  return li;
}

function renderDocuments(docs: DocumentInfo[]) {
  documentsCount.textContent = docs.length ? plural(docs.length, "file") : "";
  documentsEl.replaceChildren(...docs.map((doc) => documentRow(doc)));
  if (docs.length === 0) {
    setStatus("No documents yet. Drop a PDF or text file above to get started.");
  }
}

async function refreshDocuments() {
  try {
    renderDocuments(await listDocuments());
  } catch (err) {
    setStatus(err instanceof Error ? err.message : "Could not load documents.", "error");
  }
}

async function removeDocument(source: string, button: HTMLButtonElement) {
  button.disabled = true;
  try {
    await deleteDocument(source);
    forgetCitation(source);
    setStatus(`Removed ${source}.`);
    await refreshDocuments();
  } catch (err) {
    button.disabled = false;
    setStatus(err instanceof Error ? err.message : "Could not remove document.", "error");
  }
}

const ACCEPTED = /\.(pdf|txt)$/i;

async function uploadFiles(files: FileList | File[]) {
  const list = Array.from(files);
  if (list.length === 0) return;

  const rejected = list.filter((f) => !ACCEPTED.test(f.name));
  const accepted = list.filter((f) => ACCEPTED.test(f.name));
  if (accepted.length === 0) {
    setStatus("Only PDF and TXT files are supported.", "error");
    return;
  }

  dropzone.classList.add("is-busy");
  for (const file of accepted) {
    documentsEl.querySelector(`[data-source="${CSS.escape(file.name)}"]`)?.remove();
    documentsEl.prepend(documentRow({ source: file.name, chunks: 0 }, true));
  }

  const failures: string[] = [];
  let indexed = 0;
  for (const file of accepted) {
    setStatus(`Indexing ${file.name}…`);
    try {
      await uploadDocument(file);
      forgetCitation(file.name);
      indexed += 1;
    } catch (err) {
      failures.push(`${file.name}: ${err instanceof Error ? err.message : "upload failed"}`);
    }
  }

  dropzone.classList.remove("is-busy");
  await refreshDocuments();

  if (failures.length) {
    setStatus(failures.join(" · "), "error");
  } else {
    const skipped = rejected.length ? ` Skipped ${plural(rejected.length, "unsupported file")}.` : "";
    setStatus(`Indexed ${plural(indexed, "file")}. Ask away.${skipped}`, "success");
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files) void uploadFiles(fileInput.files);
  fileInput.value = "";
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("is-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-over"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-over");
  dropOverlay.hidden = true;
  if (event.dataTransfer?.files) void uploadFiles(event.dataTransfer.files);
});

// Whole-window drop: show an overlay while a file is dragged anywhere over
// the page, and accept the drop even if it misses the dropzone.
let dragDepth = 0;

function hasFiles(event: DragEvent) {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

window.addEventListener("dragenter", (event) => {
  if (!hasFiles(event)) return;
  dragDepth += 1;
  dropOverlay.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropOverlay.hidden = true;
});
window.addEventListener("dragover", (event) => {
  if (hasFiles(event)) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  dragDepth = 0;
  dropOverlay.hidden = true;
  if (!hasFiles(event)) return;
  event.preventDefault();
  if (event.dataTransfer?.files) void uploadFiles(event.dataTransfer.files);
});

void refreshDocuments();
