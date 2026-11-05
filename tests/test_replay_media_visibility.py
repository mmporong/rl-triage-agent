"""재생 파일 수용과 브라우저 표시를 구분하고 영상 접근 경로를 유지한다."""
from __future__ import annotations

import hashlib
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Media(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "video":
            self.videos.append(dict(attrs))
        elif tag == "a":
            self.links.append(dict(attrs))


def render(*, waiting=False):
    script = """
const fs = require('fs'), vm = require('vm');
const app = require('./docs/replay/app.js'), context = {window: {}};
vm.runInNewContext(fs.readFileSync('./docs/replay/data.js', 'utf8'), context);
const data = context.window.REPLAY_DATA;
if (process.argv[1] === 'waiting') {
  for (const block of data.scenes[0].blocks) if (block.type === 'video') {
    for (const item of block.items) data.sources[item.source].status = 'waiting';
  }
}
console.log(app.renderScene(data, 0, {protocol: 'https:', hostname: 'mmporong.github.io',
                                  pathname: '/rl-triage-agent/replay/'}));
"""
    return subprocess.run(["node", "-e", script, "waiting" if waiting else "ready"],
                          cwd=ROOT, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def test_ready_videos_have_local_preview_and_independent_playback_link():
    media = Media()
    media.feed(render())
    assert len(media.videos) == 3
    for video in media.videos:
        assert video.get("poster", "").startswith("posters/")
        assert (ROOT / "docs/replay" / video["poster"]).is_file()
        assert "playsinline" in video
        assert any(link.get("href") == video["src"] and link.get("target") == "_blank"
                   and "noopener" in link.get("rel", "") for link in media.links)


def test_unaccepted_videos_do_not_gain_a_preview_or_playback_link():
    media = Media()
    media.feed(render(waiting=True))
    assert media.videos == []
    assert not any(link.get("class") == "video-open" for link in media.links)


def test_preview_provenance_matches_unchanged_video_and_derived_image():
    folder = ROOT / "docs/replay/posters"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["frames"]) == 4
    for row in manifest["frames"]:
        assert hashlib.sha256((ROOT / row["source"]).read_bytes()).hexdigest() == row["source_sha256"]
        assert hashlib.sha256((folder / row["image"]).read_bytes()).hexdigest() == row["image_sha256"]
        assert row["timestamp_s"] == 1


def test_video_decode_error_reveals_fallback_without_changing_evidence_status():
    script = """
const assert = require('assert'), app = require('./docs/replay/app.js');
const message = {hidden: true}, video = {tagName: 'VIDEO',
  parentNode: {querySelector: () => message}};
app.showVideoError({target: {tagName: 'IMG'}});
app.showVideoError({target: video});
assert.strictEqual(message.hidden, false);
assert.strictEqual(video.evidenceStatus, undefined);
"""
    subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True,
                   text=True, encoding="utf-8", errors="replace")
