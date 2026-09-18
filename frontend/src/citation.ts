import { getCitation, type Citation } from "./api";

type Style = "ieee" | "apa";

const dialog = document.getElementById("cite-dialog") as HTMLDialogElement;
const sourceEl = document.getElementById("cite-source") as HTMLElement;
const closeButton = document.getElementById("cite-close") as HTMLButtonElement;
const tabs = Array.from(dialog.querySelectorAll<HTMLButtonElement>(".segmented button"));
const loadingEl = document.getElementById("cite-loading") as HTMLElement;
const textEl = document.getElementById("cite-text") as HTMLElement;
const errorEl = document.getElementById("cite-error") as HTMLElement;
const detailsEl = document.getElementById("cite-details") as HTMLDListElement;
const copyButton = document.getElementById("cite-copy") as HTMLButtonElement;

const cache = new Map<string, Citation>();
let current: Citation | null = null;
let style: Style = "ieee";
let copyResetTimer: number | undefined;
let requestId = 0;

function setStyle(next: Style) {
  style = next;
  for (const tab of tabs) {
    const active = tab.dataset.style === next;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  }
  render();
}

function formatAuthors(citation: Citation): string {
  const names = citation.metadata.authors.map((a) => `${a.given} ${a.family}`.trim());
  return names.length ? names.join(", ") : "Not detected";
}

function render() {
  if (!current) return;
  textEl.textContent = current[style];
  textEl.hidden = false;
  loadingEl.hidden = true;
  errorEl.hidden = true;
  copyButton.disabled = false;

  const rows: [string, string][] = [
    ["Title", current.metadata.title],
    ["Authors", formatAuthors(current)],
    ["Year", current.metadata.year ? String(current.metadata.year) : "Not detected"],
  ];
  if (current.metadata.venue) rows.push(["Source", current.metadata.venue]);
  detailsEl.replaceChildren(
    ...rows.flatMap(([label, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }),
  );
  detailsEl.hidden = false;
}

function showLoading() {
  current = null;
  loadingEl.hidden = false;
  textEl.hidden = true;
  errorEl.hidden = true;
  detailsEl.hidden = true;
  copyButton.disabled = true;
}

function showError(message: string) {
  current = null;
  loadingEl.hidden = true;
  textEl.hidden = true;
  detailsEl.hidden = true;
  errorEl.textContent = message;
  errorEl.hidden = false;
  copyButton.disabled = true;
}

export async function openCitation(source: string) {
  sourceEl.textContent = source;
  resetCopyButton();
  if (!dialog.open) dialog.showModal();

  const cached = cache.get(source);
  if (cached) {
    current = cached;
    render();
    return;
  }

  const id = ++requestId;
  showLoading();
  try {
    const citation = await getCitation(source);
    cache.set(source, citation);
    if (id !== requestId) return;
    current = citation;
    render();
  } catch (err) {
    if (id !== requestId) return;
    showError(err instanceof Error ? err.message : "Could not build a reference.");
  }
}

export function forgetCitation(source: string) {
  cache.delete(source);
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.append(scratch);
    scratch.select();
    const ok = document.execCommand("copy");
    scratch.remove();
    return ok;
  }
}

function resetCopyButton() {
  window.clearTimeout(copyResetTimer);
  copyButton.textContent = "Copy";
}

copyButton.addEventListener("click", async () => {
  if (!current) return;
  const ok = await copyText(current[style]);
  copyButton.textContent = ok ? "Copied" : "Copy failed";
  window.clearTimeout(copyResetTimer);
  copyResetTimer = window.setTimeout(resetCopyButton, 1600);
});

for (const tab of tabs) {
  tab.addEventListener("click", () => setStyle(tab.dataset.style as Style));
}

closeButton.addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});
