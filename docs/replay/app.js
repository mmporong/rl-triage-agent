/* 발표 재생 화면. data.js(window.REPLAY_DATA)만 읽고 네트워크를 쓰지 않는다. */
(function (root) {
  "use strict";

  var LINE_COLORS = ["#b4451f", "#7c3aed", "#047857"];
  // 원본 MP4의 1초 프레임. 출처·영상/이미지 SHA는 posters/manifest.json에 보존한다.
  var VIDEO_POSTERS = {
    "docs/media/go2-correction-reward_resume_dev3_20261013/comparison.mp4": "posters/reward-resume-s2027.jpg",
    "docs/media/go2-correction-reward_v4_s2026_20261013/comparison.mp4": "posters/reward-modes-s2026.jpg",
    "docs/media/go2-correction-reward_v4_s2028_20261013/comparison.mp4": "posters/reward-modes-s2028.jpg",
    "docs/media/five_type_recovery_s5557_20261102_v1/comparison.mp4": "posters/compound-recovery-s5557.jpg"
  };

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function statusLabel(data, status) {
    return (data.status_labels && data.status_labels[status]) || status;
  }

  function formatNumber(value, digits, format) {
    if (format === "sci") return value.toExponential(1);
    if (digits === null || digits === undefined) return String(value);
    return Number(value).toFixed(digits);
  }

  function collectSources(block) {
    var ids = [];
    function add(id) { if (id && ids.indexOf(id) < 0) ids.push(id); }
    (block.status_sources || []).forEach(add);
    (block.rows || []).forEach(function (row) {
      row.forEach(function (cell) { if (cell && typeof cell === "object") add(cell.source); });
    });
    (block.items || []).forEach(function (item) { add(item.source); });
    add(block.source);
    return ids;
  }

  // 근거·영상 주소. 파일로 열거나 저장소 루트를 서빙하면 저장소 상대 경로("../../" + 경로)다. GitHub Pages(main의 /docs가
  // 사이트 루트, <owner>.github.io/<repo>/replay/)에서 열면 docs/ 아래는 사이트 상대 경로, 그 밖은 같은 저장소 main의
  // GitHub 파일 화면이다. 영상은 GitHub 파일 화면에서 재생되지 않으므로 docs/ 아래가 아니면 주소를 주지 않는다(null).
  function linker(location) {
    var host = (location && location.hostname) || "";
    var pages = !!location && location.protocol === "https:" && /\.github\.io$/.test(host);
    // 프로젝트 사이트는 /<repo>/replay/, 사용자 사이트(저장소 이름이 <owner>.github.io)는 /replay/다.
    var parts = pages ? String(location.pathname || "").split("/").filter(Boolean) : [];
    var at = parts.indexOf("replay");
    var repo = at === 1 ? parts[0] : at === 0 ? host : null;
    var blob = repo ? location.protocol + "//github.com/" + host.slice(0, -".github.io".length) + "/" + repo + "/blob/main/" : null;
    return function (path, kind) {
      if (!pages) return "../../" + path;
      if (path.indexOf("docs/") === 0) return "../" + path.slice("docs/".length);
      return kind === "video" || !blob ? null : blob + encodeURI(path);
    };
  }

  function refLink(data, ids, sourceId, link) {
    var index = ids.indexOf(sourceId) + 1;
    var row = data.sources[sourceId] || {};
    var title = (row.path || "경로 미정") + (row.sha256 ? " · sha256 " + row.sha256 : "") + " · " + statusLabel(data, row.status);
    var href = row.path ? link(row.path, "file") : null;
    if (!href) return '<span class="ref" title="' + esc(title) + '">[' + index + "]</span>";
    return '<a class="ref" href="' + esc(href) + '" title="' + esc(title) + '">[' + index + "]</a>";
  }

  // 판정 문자열은 색과 글자로 함께 구분한다. 정확히 일치하는 값만 바꾸고 나머지 문구는 그대로 둔다.
  var VERDICT_TONES = {
    normal: "good", "예": "good", pass: "good", healthy: "good", recovered: "good", confirmed: "good", ready: "good",
    abnormal: "bad", "아니요": "bad", fail: "bad", unhealthy: "bad", not_recovered: "bad", invalid: "bad"
  };

  function textValue(value) {
    var tone = VERDICT_TONES[String(value).trim()];
    return tone ? '<span class="verdict ' + tone + '">' + esc(value) + "</span>" : esc(value);
  }

  function renderCell(data, ids, cell, link) {
    if (cell === null || cell === undefined) return "<td></td>";
    if (typeof cell !== "object") return "<td>" + textValue(cell) + "</td>";
    var row = data.sources[cell.source] || { status: "waiting" };
    var body;
    if (cell.value === null || cell.value === undefined) {
      var state = row.status === "ready" ? "waiting" : row.status;
      body = '<span class="pending ' + esc(state) + '">' + esc(statusLabel(data, state)) + "</span>";
    } else if (typeof cell.value === "number") {
      body = esc(formatNumber(cell.value, cell.digits, cell.format));
    } else {
      body = textValue(cell.value);
    }
    var numeric = typeof cell.value === "number" ? ' class="num"' : "";
    return "<td" + numeric + ">" + body + refLink(data, ids, cell.source, link) + "</td>";
  }

  function sourceSummary(data, ids) {
    var counts = {};
    ids.forEach(function (id) { var status = (data.sources[id] || { status: "waiting" }).status; counts[status] = (counts[status] || 0) + 1; });
    return ["ready", "waiting", "mismatch"].filter(function (status) { return counts[status]; }).map(function (status) {
      return '<span class="status ' + status + '">' + esc(statusLabel(data, status)) + " " + counts[status] + "</span>";
    }).join(" ");
  }

  function renderSources(data, ids, link) {
    if (!ids.length) return "";
    return '<details class="evidence"><summary>근거 파일 ' + ids.length + "개 " + sourceSummary(data, ids) + "</summary>" +
      '<ol class="sources">' + ids.map(function (id, index) {
      var row = data.sources[id] || { status: "waiting" };
      var badge = '<span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span>";
      var href = row.path ? link(row.path, "file") : null;
      var path = !row.path ? "경로 미정" : href ? '<a href="' + esc(href) + '">' + esc(row.path) + "</a>" : esc(row.path);
      var sha = row.sha256 ? ' <span class="sha" title="' + esc(row.sha256) + '">sha256 ' + esc(row.sha256.slice(0, 12)) + "…</span>" : "";
      return "<li>[" + (index + 1) + "] " + badge + " " + path + sha + "</li>";
    }).join("") + "</ol></details>";
  }

  function renderTable(data, block, ids, link) {
    var head = "<tr>" + block.columns.map(function (column) { return "<th>" + esc(column) + "</th>"; }).join("") + "</tr>";
    var rows = block.rows.length ? block.rows.map(function (row) {
      return "<tr>" + row.map(function (cell) { return renderCell(data, ids, cell, link); }).join("") + "</tr>";
    }).join("") : '<tr><td colspan="' + block.columns.length + '">' + pendingFor(data, ids) + "</td></tr>";
    return '<div class="table-wrap"><table><thead>' + head + "</thead><tbody>" + rows + "</tbody></table></div>";
  }

  // 표만 있고 행이 적은 장면은 글자를 키워 빈 화면을 줄인다. 밀집 장면에는 적용하지 않는다.
  function roomy(scene) {
    if (scene.dense || scene.blocks.length > 2) return false;
    return scene.blocks.every(function (block) { return block.type === "table" && block.rows.length <= 8; });
  }

  function pendingFor(data, ids) {
    var worst = "waiting";
    ids.forEach(function (id) { if ((data.sources[id] || {}).status === "mismatch") worst = "mismatch"; });
    return '<span class="pending ' + worst + '">' + esc(statusLabel(data, worst)) + "</span>";
  }

  function extent(panel) {
    var values = [];
    if (panel.envelope) values = values.concat(panel.envelope.lo, panel.envelope.hi);
    panel.lines.forEach(function (line) { values = values.concat(line.values); });
    values = values.filter(function (value) { return typeof value === "number"; });
    var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    if (lo === hi) { lo -= 1; hi += 1; }
    var pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }

  function renderPanel(panel) {
    if (!panel.envelope) return '<div class="chart"><h3>' + esc(panel.title) + '</h3><div class="placeholder">대기</div></div>';
    var width = 560, height = 250, left = 58, right = 10, top = 10, bottom = 28;
    var count = panel.envelope.lo.length, range = extent(panel);
    function x(i) { return left + (i + panel.x0) / 299 * (width - left - right); }
    function y(v) { return top + (range[1] - v) / (range[1] - range[0]) * (height - top - bottom); }
    var band = panel.envelope.lo.map(function (v, i) { return x(i).toFixed(1) + "," + y(v).toFixed(1); })
      .concat(panel.envelope.hi.slice().reverse().map(function (v, i) {
        return x(count - 1 - i).toFixed(1) + "," + y(v).toFixed(1);
      })).join(" ");
    var lines = panel.lines.map(function (line, index) {
      var points = line.values.map(function (v, i) { return x(i).toFixed(1) + "," + y(v).toFixed(1); }).join(" ");
      return '<polyline fill="none" stroke="' + LINE_COLORS[index % LINE_COLORS.length] + '" stroke-width="2" points="' + points + '"/>';
    }).join("");
    var ticks = [range[0], (range[0] + range[1]) / 2, range[1]].map(function (v) {
      var label = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
      return '<line x1="' + left + '" x2="' + (width - right) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) +
        '" stroke="#e1e5eb"/><text x="' + (left - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end" font-size="12" fill="#5b6576">' + label + "</text>";
    }).join("");
    var xticks = [0, 100, 200, 299].map(function (t) {
      return '<text x="' + x(t - panel.x0).toFixed(1) + '" y="' + (height - 8) + '" text-anchor="middle" font-size="12" fill="#5b6576">' + t + "</text>";
    }).join("");
    return '<div class="chart"><h3>' + esc(panel.title) + '</h3><svg viewBox="0 0 ' + width + " " + height +
      '" role="img" aria-label="' + esc(panel.title) + '">' + ticks + '<polygon points="' + band + '" fill="#cbd5e1" opacity="0.75"/>' +
      lines + xticks + "</svg></div>";
  }

  function renderChart(data, block) {
    var panels = block.panels.map(renderPanel).join("");
    var first = block.panels.filter(function (panel) { return panel.envelope; })[0];
    var legend = first ? '<div class="legend"><span><span class="swatch" style="background:#cbd5e1"></span>' + esc(first.envelope.label) + "</span>" +
      first.lines.map(function (line, index) {
        return '<span><span class="swatch" style="background:' + LINE_COLORS[index % LINE_COLORS.length] + '"></span>' + esc(line.label) + "</span>";
      }).join("") + "<span>가로축: 학습 iteration</span></div>" : "";
    return '<div class="charts">' + panels + "</div>" + legend;
  }

  function renderVideos(data, block, link) {
    var className = block.items.length === 1 ? "videos single-video" : "videos";
    return '<div class="' + className + '">' + block.items.map(function (item) {
      var row = data.sources[item.source] || { status: "waiting" };
      var src = row.status === "ready" && row.path ? link(row.path, "video") : null;
      var poster = src && VIDEO_POSTERS[row.path];
      var media = src
        ? '<video controls preload="metadata" playsinline' + (poster ? ' poster="' + esc(poster) + '"' : '') +
          ' aria-label="' + esc(item.label) + '" src="' + esc(src) + '"></video>'
        : '<div class="placeholder"><span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span></div>";
      var fallback = src ? '<a class="video-open" href="' + esc(src) +
        '" target="_blank" rel="noopener">영상 크게 열기 · MP4</a>' +
        '<p class="video-error" role="status" hidden>이 화면에서 재생하지 못했습니다. 위 MP4 링크로 열어 주세요.</p>' : "";
      return '<div class="video-cell">' + media + "<div>" + esc(item.label) + "</div>" + fallback + "</div>";
    }).join("") + "</div>";
  }

  function showVideoError(event) {
    if (!event.target || event.target.tagName !== "VIDEO") return;
    var message = event.target.parentNode.querySelector(".video-error");
    if (message) message.hidden = false;
  }

  function mediaTabLabel(item) {
    var seed = item.label.match(/seed (\d+)/);
    return (item.label.indexOf("여섯") >= 0 ? "회복 방식 비교" : "이어 학습 비교") +
      (seed ? " · " + seed[1] : "");
  }

  function renderFailureMedia(data, scene, link, selected) {
    var block = scene.blocks.filter(function (item) { return item.type === "video"; })[0];
    var index = Number.isInteger(selected) && selected >= 0 && selected < block.items.length ? selected : 1;
    var item = block.items[index] || block.items[0];
    var row = data.sources[item.source] || { status: "waiting" };
    var src = row.status === "ready" && row.path ? link(row.path, "video") : null;
    var poster = src && VIDEO_POSTERS[row.path];
    var tabs = block.items.map(function (candidate, i) {
      return '<button type="button" role="tab" id="media-tab-' + i + '" aria-controls="media-panel"' +
        ' aria-selected="' + (i === index) + '" tabindex="' + (i === index ? "0" : "-1") +
        '" data-media-tab="' + i + '">' + esc(mediaTabLabel(candidate)) + "</button>";
    }).join("");
    var media = src ? '<video id="featured-video" controls preload="metadata" playsinline' +
      (poster ? ' poster="' + esc(poster) + '"' : '') + ' aria-label="' + esc(item.label) +
      '" src="' + esc(src) + '"></video>' +
      '<button type="button" class="media-play" data-media-play aria-label="비교 영상 재생">' +
      '<span aria-hidden="true">▶</span><span>비교 영상 재생</span></button>' +
      '<p class="video-error" role="status" hidden>이 화면에서 재생하지 못했습니다. MP4 링크로 열어 주세요.</p>' :
      '<div class="media-unavailable"><span class="status ' + esc(row.status) + '">' +
      esc(statusLabel(data, row.status)) + '</span><p>확인된 영상이 아직 없습니다.</p></div>';
    var tools = src ? '<button type="button" class="media-expand" data-media-expand>영상 확대</button>' +
      '<a class="video-open" href="' + esc(src) + '" target="_blank" rel="noopener">원본 MP4 열기 ↗</a>' : "";
    var seed = item.label.match(/seed (\d+)/);
    var tables = scene.blocks.filter(function (candidate) { return candidate.type === "table"; });
    var values = tables[0].rows.filter(function (candidate) { return !seed || candidate[0] === seed[1]; })[0];
    var training = tables[1].rows.filter(function (candidate) { return !seed || candidate[0] === seed[1]; })[0];
    var ids = collectSources(tables[0]).concat(collectSources(tables[1]));
    var metrics = values ? '<table class="media-metrics"><caption>선속도 추종 오차 · m/s</caption>' +
      '<thead><tr><th scope="col">결함 정책</th><th scope="col">정상 정책</th></tr></thead><tbody><tr>' +
      renderCell(data, ids, values[1], link) + renderCell(data, ids, values[5], link) + '</tr></tbody></table>' : "";
    var episode = training ? '<table class="media-episode"><tbody><tr><th scope="row">결함 정책의 에피소드 길이</th>' +
      renderCell(data, ids, training[1], link) + '</tr></tbody></table>' : "";
    return '<div class="blocks media-layout"><section class="media-story" aria-label="실제 시뮬레이션 비교">' +
      '<div class="media-toolbar"><h2>같은 재생 조건, 다른 회복 방식</h2><div>' + tools + '</div></div>' +
      '<div class="media-tabs" role="tablist" aria-label="비교 영상 선택">' + tabs + '</div>' +
      '<div id="media-panel" role="tabpanel" aria-labelledby="media-tab-' + index + '">' +
      '<div class="media-screen">' + media + '</div><p class="media-caption">' + esc(item.label) +
      '</p></div></section><aside class="media-reading" aria-label="영상과 함께 볼 평가 결과">' +
      '<p class="media-eyebrow">고정 평가 · 26개 속도 명령</p><h2>버티는 것과<br>걷는 것은 다릅니다.</h2>' +
      '<p class="media-reading-lead">에피소드가 길어도 이동 명령을 제대로 따르는 것은 아닙니다.</p>' +
      '<p class="media-seed">' + esc(seed ? "seed " + seed[1] : "선택한 실행") + ' 수치 비교</p>' + metrics +
      '<p class="media-hint">오차가 작을수록 명령을 잘 따라갑니다.</p>' + episode +
      '<p class="media-verdict-note">영상은 동작 관찰용입니다.<br>회복 판정은 고정 평가 수치로 합니다.</p>' +
      '<details class="media-details"><summary>전체 수치와 근거 보기</summary><div class="media-detail-body">' +
      tables.map(function (table) { return renderBlock(data, table, link); }).join("") +
      renderSources(data, collectSources(block), link) + '</div></details></aside></div>';
  }

  function renderSlot(data, block) {
    var row = data.sources[block.source] || { status: "waiting" };
    return '<div class="slot"><span class="status ' + esc(row.status) + '">' + esc(statusLabel(data, row.status)) + "</span><span>" +
      esc(block.label) + "</span></div>";
  }

  function renderBlock(data, block, link) {
    var ids = collectSources(block);
    var body;
    if (block.type === "table") body = renderTable(data, block, ids, link);
    else if (block.type === "chart") body = renderChart(data, block);
    else if (block.type === "video") body = renderVideos(data, block, link);
    else if (block.type === "slot") body = renderSlot(data, block);
    else if (block.type === "commands") body = "<pre>" + block.lines.map(esc).join("\n") + "</pre>";
    else throw new Error("unknown block type: " + block.type);
    var title = block.title ? "<h2>" + esc(block.title) + "</h2>" : "";
    return '<section class="block' + (block.wide ? " wide" : "") + '">' + title + body + renderSources(data, ids, link) + "</section>";
  }

  // location을 주지 않으면(node 렌더 시험 등) 파일로 열 때와 같은 저장소 상대 경로를 쓴다.
  function renderScene(data, index, location, selectedVideo) {
    var link = linker(location);
    var scene = data.scenes[index];
    var total = data.scenes.length;
    var dots = data.scenes.map(function (item, i) {
      return '<a href="#' + (i + 1) + '" class="' + (i === index ? "on" : "") + '" aria-label="장면 ' + (i + 1) + '"' +
        (i === index ? ' aria-current="step"' : "") + "></a>";
    }).join("");
    var mediaFirst = scene.id === "failure";
    var body = mediaFirst ? renderFailureMedia(data, scene, link, selectedVideo) :
      '<div class="blocks' + (scene.blocks.length === 1 ? " single" : "") + (scene.dense ? " dense" : "") + (roomy(scene) ? " roomy" : "") + '">' +
      scene.blocks.map(function (block) { return renderBlock(data, block, link); }).join("") + "</div>";
    var notes = '<ul class="notes">' + (scene.notes || []).map(function (note) { return "<li>" + esc(note) + "</li>"; }).join("") + "</ul>";
    if (mediaFirst) notes = '<details class="scene-notes"><summary>관찰 범위와 판정 기준</summary>' + notes + '</details>';
    var title = mediaFirst ? '<h1 aria-label="' + esc(scene.title) + '">' + esc(scene.title.replace(/^실패 입력: /, "")) + '</h1>' :
      '<h1>' + esc(scene.title) + '</h1>';
    return '<div class="topline"><span>RL Triage · 발표 재생</span><span class="counter">' + scene.number + " / " + total + "</span></div>" +
      "<header>" + title + '<p class="lead">' + esc(scene.lead) + "</p></header>" +
      body + '<div>' + notes +
      "<footer><span>" + esc(scene.scope || data.scope) + "</span>" +
      '<nav aria-label="장면 이동"><button type="button" data-step="-1"' + (index === 0 ? " disabled" : "") + ">이전 장면</button>" +
      '<div class="dots">' + dots + "</div>" +
      '<button type="button" data-step="1"' + (index === total - 1 ? " disabled" : "") + ">다음 장면</button></nav></footer></div>";
  }

  function sceneIndex(data, hash) {
    var number = parseInt(String(hash || "").replace("#", ""), 10);
    if (!(number >= 1 && number <= data.scenes.length)) return 0;
    return number - 1;
  }

  function start(window) {
    var document = window.document, stage = document.getElementById("stage");
    var data = window.REPLAY_DATA;
    var mediaChoices = {};
    if (!data || !data.scenes) {
      stage.innerHTML = '<p class="boot">data.js가 없다. <code>evals/build_replay_page.py</code>를 실행한다.</p>';
      return;
    }
    // 좁은 화면에서 1920×1080 무대를 줄이면 글자가 읽히지 않으므로, 무대 대신 세로로 흐르는 읽기 모드를 쓴다.
    function fit() {
      var reflow = window.innerWidth < 900;
      document.documentElement.classList.toggle("reflow", reflow);
      if (reflow) {
        stage.style.transform = stage.style.left = stage.style.top = "";
        return;
      }
      var scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
      stage.style.transform = "scale(" + scale + ")";
      stage.style.left = ((window.innerWidth - 1920 * scale) / 2) + "px";
      stage.style.top = ((window.innerHeight - 1080 * scale) / 2) + "px";
    }
    function show() {
      stage.querySelectorAll("video").forEach(function (video) { video.pause(); });
      var index = sceneIndex(data, window.location.hash);
      stage.classList.toggle("media-first", data.scenes[index].id === "failure");
      stage.innerHTML = renderScene(data, index, window.location, mediaChoices[index]);
      // 큰 글자로 그린 장면이 무대를 넘치면 기본 크기로 되돌린다.
      var blocks = stage.querySelector(".blocks.roomy");
      if (blocks && blocks.scrollHeight > blocks.clientHeight + 1) blocks.classList.remove("roomy");
    }
    function chooseMedia(index, focus) {
      mediaChoices[sceneIndex(data, window.location.hash)] = index;
      show();
      if (focus) stage.querySelector('[data-media-tab="' + index + '"]').focus();
    }
    function expandMedia(video, trigger) {
      var playing = !video.paused, time = video.currentTime;
      video.pause();
      var dialog = document.createElement("dialog");
      dialog.className = "media-dialog";
      dialog.setAttribute("aria-label", "비교 영상 확대 재생");
      dialog.innerHTML = '<div class="media-dialog-bar"><span>실제 시뮬레이션 비교</span><button type="button">닫기 · Esc</button></div>';
      var expanded = video.cloneNode(true);
      expanded.removeAttribute("id");
      dialog.appendChild(expanded);
      document.body.appendChild(dialog);
      expanded.addEventListener("loadedmetadata", function () {
        expanded.currentTime = time;
        if (playing) expanded.play().catch(function () { /* 재생 권한 거부 시 기본 제어를 유지한다. */ });
      }, { once: true });
      dialog.querySelector("button").addEventListener("click", function () { dialog.close(); });
      dialog.addEventListener("click", function (event) { if (event.target === dialog) dialog.close(); });
      dialog.addEventListener("close", function () {
        expanded.pause();
        if (Number.isFinite(expanded.currentTime)) video.currentTime = expanded.currentTime;
        dialog.remove();
        trigger.focus();
      });
      dialog.showModal();
    }
    function go(step) {
      var next = Math.max(0, Math.min(data.scenes.length - 1, sceneIndex(data, window.location.hash) + step));
      window.location.hash = "#" + (next + 1);
    }
    stage.addEventListener("click", function (event) {
      var tab = event.target.closest && event.target.closest("[data-media-tab]");
      if (tab) { chooseMedia(Number(tab.getAttribute("data-media-tab")), true); return; }
      var play = event.target.closest && event.target.closest("[data-media-play]");
      var expand = event.target.closest && event.target.closest("[data-media-expand]");
      var video = stage.querySelector("#featured-video");
      if (play && video) {
        video.play().catch(function () { showVideoError({ target: video }); });
        return;
      }
      if (expand && video) { expandMedia(video, expand); return; }
      var step = event.target.getAttribute && event.target.getAttribute("data-step");
      if (step) go(parseInt(step, 10));
    });
    // error는 버블링하지 않는다. 파일 수용 상태와 브라우저 디코딩 실패를 별도로 표시한다.
    stage.addEventListener("error", showVideoError, true);
    ["play", "pause", "ended"].forEach(function (name) {
      stage.addEventListener(name, function (event) {
        if (event.target.id !== "featured-video") return;
        var play = stage.querySelector("[data-media-play]");
        if (play) play.hidden = !event.target.paused;
      }, true);
    });
    window.addEventListener("keydown", function (event) {
      if (document.querySelector(".media-dialog[open]")) return;
      var target = event.target;
      var tab = target.closest && target.closest("[data-media-tab]");
      if (tab && ["ArrowRight", "ArrowLeft", "Home", "End"].indexOf(event.key) >= 0) {
        event.preventDefault();
        var tabs = stage.querySelectorAll("[data-media-tab]");
        var current = Number(tab.getAttribute("data-media-tab"));
        var next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 :
          (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        chooseMedia(next, true);
        return;
      }
      if ((target.closest && target.closest("button, a, video, summary, input, select, textarea")) || target.isContentEditable) return;
      if (["ArrowRight", "PageDown", " "].indexOf(event.key) >= 0) { event.preventDefault(); go(1); }
      if (["ArrowLeft", "PageUp"].indexOf(event.key) >= 0) { event.preventDefault(); go(-1); }
      if (event.key === "Home") window.location.hash = "#1";
      if (event.key === "End") window.location.hash = "#" + data.scenes.length;
    });
    window.addEventListener("hashchange", show);
    window.addEventListener("resize", fit);
    fit();
    show();
  }

  var api = { esc: esc, formatNumber: formatNumber, collectSources: collectSources, linker: linker, renderScene: renderScene,
              sceneIndex: sceneIndex, showVideoError: showVideoError };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root && root.document) start(root);
})(typeof window !== "undefined" ? window : null);
