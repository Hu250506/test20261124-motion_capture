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
      datasets: [{ label: "예측 건수", data: data.actions.counts, backgroundColor: color("--accent"), borderRadius: 4, maxBarThickness: 28 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { grid: { display: false } } },
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
