// Django가 넘겨준 ORM 집계 값으로 Chart.js 그래프 2개를 그린다.
(function () {
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  Chart.defaults.color = color("--muted");
  Chart.defaults.borderColor = color("--line");

  new Chart(document.getElementById("actionChart"), {
    type: "bar",
    data: {
      labels: data.actions.labels,
      datasets: [{ label: "예측 건수", data: data.actions.counts, backgroundColor: color("--accent"), borderRadius: 4, maxBarThickness: 18 }],
    },
    options: {
      indexAxis: "y", // 동작 이름이 길어서 가로 막대로 표시
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, ticks: { precision: 0 } }, y: { grid: { display: false }, ticks: { autoSkip: false } } },
    },
  });

  const statusColors = { 대기: color("--pending"), "처리 중": color("--processing"), 완료: color("--ok"), 실패: color("--bad") };
  new Chart(document.getElementById("statusChart"), {
    type: "doughnut",
    data: {
      labels: data.statuses.labels,
      datasets: [{ data: data.statuses.counts, backgroundColor: data.statuses.labels.map((l) => statusColors[l]), borderWidth: 2, borderColor: color("--card") }],
    },
    options: { maintainAspectRatio: false, cutout: "55%", plugins: { legend: { position: "bottom" } } },
  });
})();
