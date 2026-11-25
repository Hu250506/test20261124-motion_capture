// 분석 화면: 작업 등록(POST) → 작업 ID로 상태 조회(2초마다) → 완료되면 결과 표시
(function () {
  const API = "/api/v1/analyses/";
  const form = document.getElementById("analysis-form");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;
  const errorBox = document.getElementById("form-error");
  const startBtn = document.getElementById("start-btn");
  const current = document.getElementById("current");
  let pollTimer = null;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString("ko-KR", { dateStyle: "short", timeStyle: "short" }) : "");
  const badge = (s, label) => `<span class="badge ${s.toLowerCase()}">${esc(label)}</span>`;

  // ── 입력 방식 탭 ──
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      const mode = tab.dataset.mode;
      form.input_type.value = mode;
      document.querySelectorAll(".mode").forEach((m) => (m.hidden = !m.classList.contains("mode-" + mode)));
    })
  );

  // ── 작업 등록 ──
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    const data = new FormData();
    const mode = form.input_type.value;
    data.append("input_type", mode);
    data.append("title", form.title.value);
    data.append("expected_action", form.expected_action.value);
    if (mode === "UPLOAD") {
      if (!form.video.files.length) return showError("영상 파일을 선택하세요.");
      data.append("video", form.video.files[0]);
    } else {
      if (!form.camera_name) return showError("설정된 카메라가 없습니다.");
      data.append("camera_name", form.camera_name.value);
      data.append("capture_seconds", form.capture_seconds.value);
    }
    startBtn.disabled = true;
    startBtn.textContent = "등록 중…";
    try {
      const res = await fetch(API, { method: "POST", body: data, headers: { "X-CSRFToken": csrf }, credentials: "same-origin" });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) return showError(flattenErrors(body) || `요청 실패 (HTTP ${res.status})`);
      form.reset();
      form.input_type.value = mode;
      watch(body.id);
      loadHistory();
    } catch (err) {
      showError("서버에 연결할 수 없습니다.");
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "AI 분석 시작";
    }
  });

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  }
  function flattenErrors(body) {
    if (typeof body !== "object" || !body) return "";
    return Object.values(body).flat().map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(" ");
  }

  // ── 상태 조회(폴링) ──
  function watch(id) {
    clearTimeout(pollTimer);
    current.hidden = false;
    document.getElementById("job-id").textContent = "#" + id;
    document.getElementById("result").innerHTML = "";
    const tick = async () => {
      const res = await fetch(`${API}${id}/status/`, { credentials: "same-origin" });
      if (!res.ok) return setStatus("FAILED", "상태를 불러오지 못했습니다.");
      const s = await res.json();
      setStatus(s.status, s.status_display + (s.status === "PROCESSING" ? " — 영상 확보·AI 분석 중입니다." : s.status === "PENDING" ? " — Worker가 작업을 가져가기를 기다리는 중입니다." : ""));
      if (s.is_finished) {
        showDetail(id);
        loadHistory();
      } else {
        pollTimer = setTimeout(tick, 2000);
      }
    };
    tick();
  }

  function setStatus(status, text) {
    const order = { PENDING: 0, PROCESSING: 1, COMPLETED: 2, FAILED: 2 };
    document.querySelectorAll("#steps li").forEach((li, i) => {
      li.classList.toggle("done", i < order[status] || (i === 2 && order[status] === 2));
      li.classList.toggle("now", i === order[status] && order[status] < 2);
      li.classList.toggle("failed", i === 2 && status === "FAILED");
    });
    document.getElementById("status-line").innerHTML = badge(status, status) + " " + esc(text);
  }

  async function showDetail(id) {
    const res = await fetch(`${API}${id}/`, { credentials: "same-origin" });
    if (!res.ok) return;
    const job = await res.json();
    current.hidden = false;
    document.getElementById("job-id").textContent = "#" + job.id;
    setStatus(job.status, job.status_display);
    document.getElementById("result").innerHTML = renderResult(job);
  }

  function renderResult(job) {
    if (job.status === "FAILED") {
      return `<div class="fail"><strong>분석 실패</strong><p>${esc(job.error_message)}</p><p class="muted">오류 코드: ${esc(job.error_code)}</p></div>`;
    }
    if (job.status !== "COMPLETED") return "";
    const rows = job.predictions.map((p) => {
      const pct = Math.round(p.confidence * 100);
      const top = p.top3.map((t) => `${esc(t.action_label)} ${Math.round(t.confidence * 100)}%`).join(" · ");
      return `<tr>
        <td>${p.track_id}</td>
        <td><strong>${esc(p.action_label)}</strong>${p.action_name ? " " + esc(p.action_name) : ""}
            ${p.uncertain ? '<span class="badge warn">판단 불확실</span>' : ""}</td>
        <td><div class="bar"><span style="width:${pct}%"></span></div>${pct}%</td>
        <td class="muted">${top}</td>
        <td>${p.valid_frames}</td></tr>`;
    }).join("");
    const meta = [
      job.input_type_display,
      job.video_duration_sec != null ? `영상 ${job.video_duration_sec.toFixed(1)}초` : "",
      job.ai_processing_sec != null ? `AI 추론 ${job.ai_processing_sec.toFixed(1)}초` : "",
      job.total_seconds != null ? `전체 ${job.total_seconds.toFixed(1)}초` : "",
      job.expected_action ? `수행한 동작(선택) ${esc(job.expected_action)}` : "",
    ].filter(Boolean).join(" · ");
    return `<p class="muted">${meta}</p>
      <table class="table"><thead><tr><th>Track ID</th><th>예측 동작</th><th>신뢰도</th><th>후보 Top 3</th><th>유효 Frame</th></tr></thead>
      <tbody>${rows}</tbody></table>
      ${job.video_url ? `<video src="${esc(job.video_url)}" controls muted playsinline class="preview"></video>` : ""}
      <p class="muted small">예측은 학습한 17개 동작 중 하나로만 나옵니다. 학습하지 않은 동작도 그중 하나로 표시될 수 있으니 신뢰도와 함께 보세요.</p>`;
  }

  // ── 최근 이력 ──
  async function loadHistory() {
    const tbody = document.querySelector("#history tbody");
    const res = await fetch(API, { credentials: "same-origin" });
    if (!res.ok) return (tbody.innerHTML = `<tr><td colspan="7" class="error">이력을 불러오지 못했습니다.</td></tr>`);
    const page = await res.json();
    if (!page.results.length) return (tbody.innerHTML = `<tr><td colspan="7" class="muted">아직 분석 기록이 없습니다.</td></tr>`);
    tbody.innerHTML = page.results.map((j) => {
      const preds = j.predictions.map((p) => `T${p.track_id}:${esc(p.action_label)} ${Math.round(p.confidence * 100)}%`).join(", ");
      return `<tr>
        <td>${j.id}</td><td>${esc(j.title) || '<span class="muted">-</span>'}</td><td>${esc(j.input_type_display)}</td>
        <td>${badge(j.status, j.status_display)}</td><td>${preds || (j.status === "FAILED" ? esc(j.error_code) : "")}</td>
        <td>${fmtTime(j.created_at)}</td>
        <td class="actions"><button class="link" data-view="${j.id}">보기</button><button class="link danger" data-del="${j.id}">삭제</button></td></tr>`;
    }).join("");
  }

  document.getElementById("history").addEventListener("click", async (e) => {
    const view = e.target.dataset.view, del = e.target.dataset.del;
    if (view) {
      const res = await fetch(`${API}${view}/status/`, { credentials: "same-origin" });
      const s = await res.json();
      s.is_finished ? showDetail(view) : watch(view);
      current.scrollIntoView({ behavior: "smooth" });
    }
    if (del && confirm(`#${del} 분석 기록과 사람별 예측, 영상 파일을 삭제할까요?`)) {
      const res = await fetch(`${API}${del}/`, { method: "DELETE", headers: { "X-CSRFToken": csrf }, credentials: "same-origin" });
      if (!res.ok) alert((await res.json().catch(() => ({}))).detail || "삭제하지 못했습니다.");
      loadHistory();
    }
  });

  loadHistory();
})();
