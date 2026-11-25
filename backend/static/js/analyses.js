// 분석 화면: 작업 등록(POST) → 작업 ID로 상태 조회(2초마다) → 완료되면 결과 표시
(function () {
  const ACTION_NAMES = JSON.parse(document.getElementById("action-names").textContent);
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
      mode === "CAMERA" ? startPreview() : stopPreview();
    })
  );

  // ── 카메라 미리보기: 서버를 거쳐 현재 화면을 계속 다시 받아 온다 ──
  // 브라우저는 카메라 주소를 모르고 이름만 보낸다. 한 장을 다 그리면 다음 장을 요청한다.
  // '관절 표시'를 켜면 같은 사진의 관절 좌표(FastAPI YOLO Pose)도 받아 그 위에 그린다.
  const canvas = document.getElementById("camera-preview");
  const previewStatus = document.getElementById("camera-preview-status");
  const poseToggle = document.getElementById("pose-toggle");
  // COCO 17 관절: 0 코 1 왼눈 2 오른눈 3 왼귀 4 오른귀 5 왼어깨 6 오른어깨 7 왼팔꿈치 8 오른팔꿈치
  // 9 왼손목 10 오른손목 11 왼골반 12 오른골반 13 왼무릎 14 오른무릎 15 왼발목 16 오른발목
  const BONES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
  const COLORS = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#a78bfa", "#fb7185"];
  const KP_MIN_CONF = 0.3; // 학습 때와 같은 관절 신뢰도 기준
  const POSE_STATUS = {
    analysis_running: "영상 분석 중이라 관절 표시를 잠시 멈췄습니다.",
    pose_busy: "관절 검출 요청이 밀려 있습니다.",
    ai_unavailable: "AI 서비스에 연결할 수 없어 관절을 표시하지 못했습니다.",
  };
  let previewOn = false;
  let previewTimer = null;

  const loadImage = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

  function drawPersons(ctx, persons, w) {
    const lw = Math.max(2, w / 320);
    persons.forEach((p, i) => {
      const color = COLORS[i % COLORS.length];
      const kp = p.keypoints;
      const ok = (j) => kp[j] && kp[j][2] >= KP_MIN_CONF;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = lw;
      const [x1, y1, x2, y2] = p.box;
      ctx.setLineDash([lw * 3, lw * 2]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.font = `bold ${Math.round(lw * 7)}px sans-serif`;
      ctx.fillText(`사람 ${i + 1}`, x1 + lw, Math.max(lw * 8, y1 - lw * 2));
      BONES.forEach(([a, b]) => {
        if (!ok(a) || !ok(b)) return;
        ctx.beginPath();
        ctx.moveTo(kp[a][0], kp[a][1]);
        ctx.lineTo(kp[b][0], kp[b][1]);
        ctx.stroke();
      });
      kp.forEach((_, j) => {
        if (!ok(j)) return;
        ctx.beginPath();
        ctx.arc(kp[j][0], kp[j][1], lw * 1.6, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }

  async function tick() {
    const name = form.camera_name && form.camera_name.value;
    if (!previewOn) return;
    if (!name || document.hidden) return schedule(500);
    const base = `/cameras/${encodeURIComponent(name)}/`;
    try {
      let img, persons = null, poseStatus = "";
      if (poseToggle.checked) {
        const res = await fetch(base + "pose.json", { credentials: "same-origin", cache: "no-store" });
        if (!res.ok) throw new Error("camera");
        const data = await res.json();
        img = await loadImage(data.image);
        persons = data.persons;
        poseStatus = data.pose_status;
      } else {
        img = await loadImage(base + "snapshot.jpg?t=" + Date.now());
      }
      if (!previewOn) return;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      if (persons) drawPersons(ctx, persons, img.naturalWidth);
      canvas.hidden = false;
      previewStatus.textContent = !poseToggle.checked
        ? "실시간 미리보기 (분석에는 쓰이지 않음)"
        : persons
          ? `실시간 미리보기 · 관절 표시 중 (검출된 사람 ${persons.length}명, 분석에는 쓰이지 않음)`
          : POSE_STATUS[poseStatus] || `관절을 표시하지 못했습니다 (${poseStatus}).`;
      schedule(poseStatus === "analysis_running" ? 1000 : 100);
    } catch (err) {
      if (!previewOn) return;
      previewStatus.textContent = "카메라에 연결할 수 없습니다. 휴대폰 IP Webcam 앱이 켜져 있는지 확인하세요. 3초 뒤 다시 시도합니다.";
      canvas.hidden = true;
      schedule(3000);
    }
  }
  function schedule(delay) {
    clearTimeout(previewTimer);
    if (previewOn) previewTimer = setTimeout(tick, delay);
  }
  function startPreview() {
    if (!canvas || previewOn) return;
    previewOn = true;
    schedule(0);
  }
  function stopPreview() {
    previewOn = false;
    clearTimeout(previewTimer);
  }
  if (canvas) {
    form.camera_name && form.camera_name.addEventListener("change", () => schedule(0));
    poseToggle.addEventListener("change", () => schedule(0));
  }

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
      const top = p.top3.map((t) => `${esc(t.action_label)} ${esc(ACTION_NAMES[t.action_label] || "")} ${Math.round(t.confidence * 100)}%`).join(" · ");
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
      job.expected_action ? `수행한 동작(선택) ${esc(job.expected_action)} ${esc(ACTION_NAMES[job.expected_action] || "")}` : "",
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
      const preds = j.predictions.map((p) => `T${p.track_id}: ${esc(p.action_label)} ${esc(p.action_name)} ${Math.round(p.confidence * 100)}%`).join(", ");
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
