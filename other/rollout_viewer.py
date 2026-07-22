#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rollout Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a0d12;
      --panel: #10151d;
      --panel-raised: #151b25;
      --panel-soft: #0d1219;
      --line: #27303d;
      --line-strong: #374355;
      --text: #edf1f7;
      --muted: #98a5b7;
      --faint: #69778a;
      --blue: #62a9ff;
      --blue-soft: #13273d;
      --amber: #f0b75b;
      --amber-soft: #332714;
      --green: #5bd39b;
      --green-soft: #123125;
      --red: #ff7b83;
      --red-soft: #37191e;
      --purple: #bc9cff;
      --gray-soft: #202733;
      --shadow: 0 12px 28px rgba(0, 0, 0, .22);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.45;
    }
    button, input, select { font: inherit; }
    button { color: inherit; }
    .topbar {
      height: 56px;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: #0d1118;
    }
    .brand { display: flex; align-items: center; gap: 9px; flex: 0 0 auto; font-weight: 750; }
    .brand-mark {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border: 1px solid #3b526d;
      border-radius: 8px;
      background: linear-gradient(145deg, #17304a, #111b28);
      color: var(--blue);
      font-family: var(--mono);
      font-size: 12px;
    }
    .file-path {
      min-width: 0;
      flex: 1 1 auto;
      overflow: hidden;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 12px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .top-counts { display: flex; gap: 7px; flex: 0 0 auto; }
    .top-pill {
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .workspace {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(420px, 1.45fr) minmax(330px, .95fr);
      overflow: hidden;
    }
    .workspace-panel { min-width: 0; min-height: 0; background: var(--panel); }
    .left-panel {
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
    }
    .conversation-panel {
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--line);
      background: var(--bg);
    }
    .overview-panel { overflow-y: auto; overflow-x: hidden; }
    .panel-heading {
      min-height: 57px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .panel-title { margin: 0; font-size: 15px; font-weight: 750; letter-spacing: .01em; }
    .panel-subtitle { margin-top: 2px; color: var(--muted); font-size: 11px; }
    .index-count { color: var(--muted); font-family: var(--mono); font-size: 12px; white-space: nowrap; }
    .filters {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .wide { grid-column: 1 / -1; }
    .field { position: relative; min-width: 0; }
    .field input, .field select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      outline: none;
      background: #0c1118;
      color: var(--text);
      padding: 0 9px;
      font-size: 12px;
    }
    .field input:focus, .field select:focus { border-color: #4d78a8; box-shadow: 0 0 0 2px rgba(98, 169, 255, .1); }
    .search input { padding-left: 30px; }
    .search-icon { position: absolute; left: 10px; top: 8px; color: var(--faint); pointer-events: none; }
    .filter-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .control-button, .icon-button, .page-button {
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-raised);
      cursor: pointer;
    }
    .control-button { height: 32px; color: var(--muted); font-size: 12px; }
    .control-button:hover, .icon-button:hover, .page-button:hover:not(:disabled) { border-color: var(--line-strong); color: var(--text); background: #1a2230; }
    .rollout-list { min-height: 0; flex: 1 1 auto; overflow-y: auto; overflow-x: hidden; }
    .rollout-row {
      width: 100%;
      display: block;
      padding: 11px 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-left: 3px solid transparent;
      border-radius: 0;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .rollout-row:hover { background: #141b25; }
    .rollout-row.active { border-left-color: var(--blue); background: #152130; }
    .row-top { display: flex; align-items: center; gap: 7px; min-width: 0; }
    .rollout-id { min-width: 0; flex: 1 1 auto; overflow: hidden; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
    .reward-badge, .env-chip, .status-chip, .verdict-pill, .metric-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      white-space: nowrap;
    }
    .reward-badge { min-width: 46px; padding: 2px 7px; font-family: var(--mono); font-size: 11px; font-weight: 700; }
    .reward-high { border-color: #286d51; background: var(--green-soft); color: var(--green); }
    .reward-mid { border-color: #765a26; background: var(--amber-soft); color: var(--amber); }
    .reward-low { border-color: #78333b; background: var(--red-soft); color: var(--red); }
    .reward-none { background: var(--gray-soft); color: var(--muted); }
    .row-meta { display: flex; align-items: center; gap: 6px; margin-top: 7px; min-width: 0; }
    .env-chip { max-width: 72%; padding: 2px 7px; overflow: hidden; background: #192230; color: #b7c8dc; font-size: 10px; text-overflow: ellipsis; }
    .step-label { margin-left: auto; color: var(--faint); font-family: var(--mono); font-size: 10px; white-space: nowrap; }
    .preview {
      display: -webkit-box;
      margin-top: 7px;
      overflow: hidden;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .pagination {
      min-height: 47px;
      display: grid;
      grid-template-columns: 34px 1fr 34px;
      align-items: center;
      gap: 7px;
      padding: 7px 10px;
      border-top: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .page-button { height: 32px; }
    .page-button:disabled { opacity: .35; cursor: default; }
    .page-label { color: var(--muted); font-family: var(--mono); font-size: 11px; text-align: center; }
    .conversation-actions { display: flex; gap: 6px; }
    .conversation-actions .control-button { padding: 0 9px; }
    .chat-scroll { min-height: 0; flex: 1 1 auto; overflow-y: auto; overflow-x: hidden; padding: 18px; }
    .turn-card {
      margin: 0 auto 14px;
      max-width: 920px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 9px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .turn-card.role-user { border-left-color: var(--blue); }
    .turn-card.role-assistant { border-left-color: var(--amber); }
    .turn-card.role-system { border-left-color: #7c8798; }
    .turn-card.role-tool { border-left-color: var(--purple); }
    .turn-card.role-unknown { border-left-color: var(--faint); }
    .turn-head {
      min-height: 42px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 9px 7px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-raised);
    }
    .turn-card.collapsed .turn-head { border-bottom: 0; }
    .role-label { font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .role-user .role-label { color: var(--blue); }
    .role-assistant .role-label { color: var(--amber); }
    .role-system .role-label { color: #aab4c2; }
    .role-tool .role-label { color: var(--purple); }
    .turn-source { color: var(--faint); font-size: 10px; }
    .turn-spacer { flex: 1 1 auto; }
    .icon-button {
      width: 30px;
      height: 28px;
      display: grid;
      place-items: center;
      padding: 0;
      color: var(--muted);
      font-size: 12px;
    }
    .chevron { transition: transform .16s ease; }
    .turn-card.collapsed .chevron { transform: rotate(-90deg); }
    .turn-card.collapsed .turn-body { display: none; }
    .turn-body { max-height: min(68vh, 820px); overflow: auto; padding: 14px 15px; }
    .reasoning-block, .tool-block {
      margin: 0 0 14px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel-soft);
    }
    details > summary { cursor: pointer; user-select: none; }
    .reasoning-block summary, .tool-block summary, .raw-json summary, .judge-raw summary {
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .reasoning-content {
      padding: 11px 12px;
      border-top: 1px solid var(--line);
      color: #aeb8c7;
      font-style: italic;
    }
    .answer-label { margin: 0 0 7px; color: var(--faint); font-size: 10px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }
    .tool-content { padding: 10px; border-top: 1px solid var(--line); }
    .tool-call { margin-bottom: 9px; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: #0a0f15; }
    .tool-call:last-child { margin-bottom: 0; }
    .tool-name { padding: 7px 9px; border-bottom: 1px solid var(--line); color: var(--purple); font-family: var(--mono); font-size: 11px; font-weight: 700; }
    .tool-call pre {
      max-width: 100%;
      margin: 0;
      overflow: auto;
      padding: 9px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .markdown { min-width: 0; overflow-wrap: anywhere; }
    .markdown > :first-child { margin-top: 0; }
    .markdown > :last-child { margin-bottom: 0; }
    .markdown h1, .markdown h2, .markdown h3, .markdown h4, .markdown h5, .markdown h6 { margin: 1em 0 .45em; line-height: 1.25; }
    .markdown h1 { font-size: 1.45em; }
    .markdown h2 { font-size: 1.28em; }
    .markdown h3 { font-size: 1.14em; }
    .markdown p { margin: .7em 0; }
    .markdown ul, .markdown ol { margin: .65em 0; padding-left: 1.55em; }
    .markdown li { margin: .25em 0; }
    .markdown a { color: var(--blue); text-decoration: none; }
    .markdown a:hover { text-decoration: underline; }
    .markdown code { padding: 1px 5px; border: 1px solid var(--line); border-radius: 4px; background: #090d12; color: #d4deeb; font-family: var(--mono); font-size: .9em; }
    .markdown pre, pre.raw-text, .raw-json pre, .judge-raw pre {
      max-width: 100%;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.5;
    }
    .markdown pre { margin: .8em 0; padding: 11px; border: 1px solid var(--line); border-radius: 7px; background: #080c11; white-space: pre; overflow-wrap: normal; }
    .markdown pre code { padding: 0; border: 0; background: transparent; }
    .overview-content { padding: 15px; }
    .overview-section { margin-bottom: 18px; }
    .section-title { margin: 0 0 9px; color: #dfe6ef; font-size: 12px; font-weight: 800; letter-spacing: .055em; text-transform: uppercase; }
    .reward-card {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-raised);
      text-align: center;
    }
    .reward-label { color: var(--muted); font-size: 10px; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
    .reward-value { margin: 3px 0 9px; font-family: var(--mono); font-size: 38px; font-weight: 800; line-height: 1; }
    .reward-value.reward-high { background: transparent; color: var(--green); }
    .reward-value.reward-mid { background: transparent; color: var(--amber); }
    .reward-value.reward-low { background: transparent; color: var(--red); }
    .reward-value.reward-none { background: transparent; color: var(--muted); }
    .reward-parts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .reward-part { min-width: 0; padding: 7px 5px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel-soft); }
    .reward-part span { display: block; overflow: hidden; color: var(--faint); font-size: 9px; text-overflow: ellipsis; text-transform: uppercase; }
    .reward-part b { display: block; margin-top: 2px; overflow: hidden; font-family: var(--mono); font-size: 12px; text-overflow: ellipsis; }
    .formula-card { padding: 11px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-soft); }
    .formula-explanation { margin-bottom: 9px; color: var(--muted); font-size: 12px; }
    .formula-line { overflow-wrap: anywhere; color: #dce5ef; font-family: var(--mono); font-size: 11px; line-height: 1.65; }
    .formula-line.plugged { margin-top: 4px; color: var(--amber); }
    .verification-note { margin-top: 9px; padding-top: 8px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
    .metric-group { margin-bottom: 12px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; }
    .metric-group-title { padding: 7px 9px; border-bottom: 1px solid var(--line); background: var(--panel-raised); color: var(--muted); font-size: 11px; font-weight: 750; }
    .metric-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; min-height: 34px; padding: 6px 9px; border-bottom: 1px solid var(--line); }
    .metric-row:last-child { border-bottom: 0; }
    .metric-name { min-width: 0; color: #b9c3d1; font-size: 11px; overflow-wrap: anywhere; }
    .metric-value { color: var(--text); font-family: var(--mono); font-size: 11px; text-align: right; }
    .metric-pill { gap: 4px; padding: 1px 6px; font-family: var(--sans); font-size: 10px; font-weight: 700; }
    .metric-pill.good, .verdict-pill.good { border-color: #286d51; background: var(--green-soft); color: var(--green); }
    .metric-pill.bad, .verdict-pill.bad { border-color: #78333b; background: var(--red-soft); color: var(--red); }
    .metric-pill.warn, .verdict-pill.warn { border-color: #765a26; background: var(--amber-soft); color: var(--amber); }
    .metric-pill.neutral, .verdict-pill.neutral { background: var(--gray-soft); color: var(--muted); }
    .status-list { display: flex; flex-wrap: wrap; gap: 7px; }
    .status-chip { position: relative; padding: 4px 8px; font-size: 10px; font-weight: 700; outline: none; }
    .status-chip.good { border-color: #286d51; background: var(--green-soft); color: var(--green); }
    .status-chip.warn { border-color: #765a26; background: var(--amber-soft); color: var(--amber); }
    .status-chip.bad { border-color: #78333b; background: var(--red-soft); color: var(--red); }
    .status-chip.neutral { background: var(--gray-soft); color: #b5c0ce; }
    .has-tooltip { position: relative; cursor: help; }
    .has-tooltip::after {
      content: attr(data-tooltip);
      position: absolute;
      right: 0;
      top: calc(100% + 7px);
      z-index: 20;
      width: min(270px, 75vw);
      padding: 8px 9px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      background: #080c12;
      color: #d7dee8;
      box-shadow: var(--shadow);
      font-family: var(--sans);
      font-size: 11px;
      font-weight: 450;
      line-height: 1.4;
      opacity: 0;
      pointer-events: none;
      transform: translateY(-3px);
      transition: opacity .12s ease, transform .12s ease;
      white-space: normal;
    }
    .has-tooltip:hover::after, .has-tooltip:focus::after { opacity: 1; transform: translateY(0); }
    .judge-card { margin-bottom: 9px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-soft); }
    .judge-head { display: flex; align-items: center; gap: 7px; padding: 8px 9px; border-bottom: 1px solid var(--line); }
    .judge-title { min-width: 0; flex: 1 1 auto; font-size: 11px; font-weight: 700; overflow-wrap: anywhere; }
    .judge-model { color: var(--faint); font-size: 10px; white-space: nowrap; }
    .verdict-pill { padding: 2px 6px; font-size: 9px; font-weight: 800; }
    .judge-error { margin: 8px 9px 0; color: var(--red); font-family: var(--mono); font-size: 10px; overflow-wrap: anywhere; }
    .judge-raw summary { padding: 8px 9px; }
    .judge-raw-content { padding: 0 9px 9px; }
    .raw-label { margin: 8px 0 4px; color: var(--faint); font-size: 9px; font-weight: 750; letter-spacing: .07em; text-transform: uppercase; }
    .judge-raw pre { max-height: 280px; margin: 0; padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: #080c11; }
    .mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .mini-card { overflow: hidden; border: 1px solid var(--line); border-radius: 8px; }
    .mini-card-title { padding: 7px 8px; border-bottom: 1px solid var(--line); background: var(--panel-raised); color: var(--muted); font-size: 10px; font-weight: 750; }
    .mini-row { display: flex; justify-content: space-between; gap: 8px; padding: 5px 8px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 10px; }
    .mini-row:last-child { border-bottom: 0; }
    .mini-row b { color: var(--text); font-family: var(--mono); font-weight: 600; }
    .raw-json { margin-top: 18px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-soft); }
    .raw-json pre { max-height: 480px; margin: 0; padding: 10px; border-top: 1px solid var(--line); background: #080c11; }
    .empty-state, .loading-state, .error-state { padding: 32px 18px; color: var(--muted); text-align: center; }
    .error-state { color: var(--red); }
    .em-dash { color: var(--faint); }
    .toast {
      position: fixed;
      left: 50%;
      bottom: 18px;
      z-index: 50;
      padding: 7px 11px;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      background: #161e29;
      box-shadow: var(--shadow);
      color: var(--text);
      font-size: 11px;
      opacity: 0;
      pointer-events: none;
      transform: translate(-50%, 8px);
      transition: opacity .15s ease, transform .15s ease;
    }
    .toast.show { opacity: 1; transform: translate(-50%, 0); }
    @media (max-width: 1180px) {
      .workspace { grid-template-columns: 290px minmax(390px, 1.3fr) minmax(300px, .9fr); }
      .top-counts { display: none; }
    }
    @media (max-width: 1020px) {
      html, body { height: auto; min-height: 100%; }
      body { overflow: auto; }
      .topbar { position: sticky; top: 0; z-index: 30; }
      .workspace { height: auto; display: block; overflow: visible; }
      .left-panel, .conversation-panel { border-right: 0; border-bottom: 1px solid var(--line); }
      .left-panel { height: 650px; }
      .conversation-panel { height: 760px; }
      .overview-panel { overflow: visible; }
      .turn-body { max-height: 560px; }
    }
    @media (max-width: 560px) {
      .topbar { padding: 0 10px; }
      .file-path { display: none; }
      .brand { flex: 1 1 auto; }
      .chat-scroll { padding: 10px; }
      .conversation-actions .control-button { padding: 0 6px; font-size: 10px; }
      .overview-content { padding: 11px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><span class="brand-mark">RL</span><span>Rollout Viewer</span></div>
    <div id="file-path" class="file-path">Loading file…</div>
    <div id="top-counts" class="top-counts"></div>
  </header>
  <div class="workspace">
    <aside class="workspace-panel left-panel">
      <div class="panel-heading">
        <div>
          <h1 class="panel-title">Rollouts</h1>
          <div id="match-summary" class="panel-subtitle">Loading…</div>
        </div>
        <div id="index-count" class="index-count">— / —</div>
      </div>
      <div class="filters">
        <div class="field search wide">
          <span class="search-icon">⌕</span>
          <input id="q" type="search" placeholder="Search previews or labels" autocomplete="off">
        </div>
        <div class="field"><select id="env" aria-label="Environment"><option value="">All environments</option></select></div>
        <div class="field">
          <select id="kind" aria-label="Rollout kind">
            <option value="">Train + eval</option>
            <option value="train">Train</option>
            <option value="eval">Eval</option>
          </select>
        </div>
        <div class="field">
          <select id="trainable" aria-label="Trainability">
            <option value="">Any trainability</option>
            <option value="true">Trainable</option>
            <option value="false">Not trainable</option>
          </select>
        </div>
        <div class="field">
          <select id="reward" aria-label="Reward range">
            <option value="">Any reward</option>
            <option value="high">Reward ≥ 0.7</option>
            <option value="mid">Reward 0.3–0.7</option>
            <option value="low">Reward &lt; 0.3</option>
            <option value="missing">Reward missing</option>
          </select>
        </div>
        <div class="filter-actions wide">
          <button id="reset-filters" class="control-button" type="button">Reset filters</button>
          <button id="refresh" class="control-button" type="button">Refresh data</button>
        </div>
      </div>
      <div id="rollout-list" class="rollout-list"><div class="loading-state">Indexing rollouts…</div></div>
      <div class="pagination">
        <button id="previous-page" class="page-button" type="button" aria-label="Previous page">←</button>
        <div id="page-label" class="page-label">—</div>
        <button id="next-page" class="page-button" type="button" aria-label="Next page">→</button>
      </div>
    </aside>
    <main class="workspace-panel conversation-panel">
      <div class="panel-heading">
        <div>
          <h2 class="panel-title">Conversation History</h2>
          <div id="conversation-summary" class="panel-subtitle">Select a rollout</div>
        </div>
        <div class="conversation-actions">
          <button id="collapse-all" class="control-button" type="button">Collapse All</button>
          <button id="expand-all" class="control-button" type="button">Expand All</button>
        </div>
      </div>
      <div id="chat" class="chat-scroll"><div class="empty-state">Select a rollout to inspect its conversation.</div></div>
    </main>
    <aside class="workspace-panel overview-panel">
      <div class="panel-heading">
        <div>
          <h2 class="panel-title">Overview</h2>
          <div id="overview-summary" class="panel-subtitle">Reward, metrics, and judges</div>
        </div>
      </div>
      <div id="overview" class="overview-content"><div class="empty-state">Select a rollout to inspect its outcome.</div></div>
    </aside>
  </div>
  <div id="toast" class="toast" role="status"></div>
<script>
"use strict";

const PAGE_SIZE = 100;
const state = {
  records: [],
  totalMatched: 0,
  offset: 0,
  selectedLine: null,
  selectedIndex: null,
  record: null,
  snapshotLine: null,
  turnCopies: new Map(),
  recordsRequest: 0,
  detailRequest: 0,
  toastTimer: null,
};
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function isPresent(value) {
  return value !== null && value !== undefined && value !== "";
}

function firstPresent(...values) {
  return values.find(isPresent);
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function toText(value) {
  if (!isPresent(value)) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function fmtNumber(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Number.isInteger(number)) return String(number);
  return number.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function fmtValue(value) {
  if (!isPresent(value)) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return fmtNumber(value);
  return String(value);
}

function rewardClass(value) {
  if (!isPresent(value) || !Number.isFinite(Number(value))) return "reward-none";
  if (Number(value) >= 0.7) return "reward-high";
  if (Number(value) >= 0.3) return "reward-mid";
  return "reward-low";
}

function safeHref(value) {
  const href = String(value ?? "").trim();
  if (/^(https?:|mailto:)/i.test(href) || /^(#|\/|\.\.?\/)/.test(href)) return href;
  return "#";
}

function renderInline(source) {
  const tokens = [];
  const keep = (html) => `\uE000${tokens.push(html) - 1}\uE001`;
  let text = String(source ?? "");
  text = text.replace(/`([^`\n]+)`/g, (_, code) => keep(`<code>${esc(code)}</code>`));
  text = text.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (_, label, href) => {
    const target = safeHref(href);
    const external = /^(https?:|mailto:)/i.test(target) ? ' target="_blank" rel="noopener noreferrer"' : "";
    return keep(`<a href="${esc(target)}"${external}>${esc(label)}</a>`);
  });
  text = esc(text);
  text = text.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  return text.replace(/\uE000(\d+)\uE001/g, (_, index) => tokens[Number(index)] ?? "");
}

function isMarkdownBlockStart(line) {
  return /^\s*```/.test(line) || /^#{1,6}\s+/.test(line) || /^\s*[-+*]\s+/.test(line) || /^\s*\d+\.\s+/.test(line);
}

function renderMarkdown(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const fence = line.match(/^\s*```\s*([^\s`]*)\s*$/);
    if (fence) {
      const language = fence[1] ? ` class="language-${esc(fence[1])}"` : "";
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      output.push(`<pre><code${language}>${esc(code.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    const unordered = line.match(/^\s*[-+*]\s+(.+)$/);
    if (unordered) {
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*[-+*]\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${renderInline(item[1])}</li>`);
        index += 1;
      }
      output.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*\d+\.\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${renderInline(item[1])}</li>`);
        index += 1;
      }
      output.push(`<ol>${items.join("")}</ol>`);
      continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    if (!paragraph.length) {
      paragraph.push(line);
      index += 1;
    }
    output.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
  }
  return output.join("");
}

async function fetchJson(url) {
  const response = await fetch(url, {cache: "no-store"});
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("show"), 1400);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const input = document.createElement("textarea");
    input.value = text;
    input.style.position = "fixed";
    input.style.opacity = "0";
    document.body.appendChild(input);
    input.select();
    document.execCommand("copy");
    input.remove();
  }
  showToast("Copied to clipboard");
}

function buildRecordQuery() {
  const params = new URLSearchParams({
    type: "rollout",
    limit: String(PAGE_SIZE),
    offset: String(state.offset),
    order: "desc",
  });
  for (const id of ["q", "env", "kind", "trainable", "reward"]) {
    const value = $(id).value.trim();
    if (value) params.set(id, value);
  }
  if (state.snapshotLine !== null) params.set("max_line", String(state.snapshotLine));
  return params.toString();
}

async function loadStats() {
  const data = await fetchJson("/api/stats");
  $("file-path").textContent = data.file || "—";
  $("file-path").title = data.file || "";
  $("top-counts").innerHTML = [
    `<span class="top-pill">${esc(fmtValue(data.rollouts))} rollouts</span>`,
    `<span class="top-pill">${esc(fmtValue(data.summaries))} summaries</span>`,
    `<span class="top-pill">${esc(fmtValue(data.total_lines))} lines</span>`,
  ].join("");
  const select = $("env");
  const selected = select.value;
  const environments = Object.entries(asObject(data.envs)).sort((a, b) => a[0].localeCompare(b[0]));
  select.innerHTML = `<option value="">All environments</option>${environments.map(([name, count]) => (
    `<option value="${esc(name)}">${esc(name)} (${esc(fmtValue(count))})</option>`
  )).join("")}`;
  if (environments.some(([name]) => name === selected)) select.value = selected;
  return data;
}

function rowIdentifier(record) {
  if (isPresent(record.example_id)) return `#${record.example_id}`;
  if (isPresent(record.rollout_id)) return `#${String(record.rollout_id).slice(0, 10)}`;
  return `line ${record.line}`;
}

function renderRolloutList() {
  const list = $("rollout-list");
  if (!state.records.length) {
    list.innerHTML = `<div class="empty-state">No rollouts match these filters.</div>`;
  } else {
    list.innerHTML = state.records.map((record, index) => {
      const active = record.line === state.selectedLine ? " active" : "";
      return `<button class="rollout-row${active}" type="button" data-line="${esc(record.line)}" data-index="${index}" aria-pressed="${active ? "true" : "false"}">
        <span class="row-top">
          <span class="rollout-id">${esc(rowIdentifier(record))}</span>
          <span class="reward-badge ${rewardClass(record.reward)}">${esc(fmtValue(record.reward))}</span>
        </span>
        <span class="row-meta">
          <span class="env-chip" title="${esc(fmtValue(record.env_name))}">${esc(fmtValue(record.env_name))}</span>
          <span class="step-label">step ${esc(fmtValue(record.step))}</span>
        </span>
        <span class="preview">${esc(record.preview || "—")}</span>
      </button>`;
    }).join("");
  }
  const start = state.totalMatched ? state.offset + 1 : 0;
  const end = Math.min(state.offset + state.records.length, state.totalMatched);
  $("match-summary").textContent = `${state.totalMatched.toLocaleString()} matching rollout${state.totalMatched === 1 ? "" : "s"}`;
  $("page-label").textContent = state.totalMatched ? `${start.toLocaleString()}–${end.toLocaleString()} of ${state.totalMatched.toLocaleString()}` : "0 of 0";
  $("previous-page").disabled = state.offset === 0;
  $("next-page").disabled = state.offset + state.records.length >= state.totalMatched;
  updateIndexCount();
}

function updateIndexCount() {
  const position = state.selectedIndex === null ? "—" : (state.offset + state.selectedIndex + 1).toLocaleString();
  $("index-count").textContent = `${position} / ${state.totalMatched.toLocaleString()}`;
}

async function loadRecords({selectFirst = true} = {}) {
  const request = ++state.recordsRequest;
  $("rollout-list").innerHTML = `<div class="loading-state">Loading rollouts…</div>`;
  try {
    const data = await fetchJson(`/api/records?${buildRecordQuery()}`);
    if (request !== state.recordsRequest) return;
    state.records = asArray(data.records);
    state.totalMatched = Number(data.total_matched) || 0;
    state.offset = Number(data.offset) || 0;
    state.snapshotLine = Number(data.snapshot_line) || null;
    const selectedIndex = state.records.findIndex((record) => record.line === state.selectedLine);
    state.selectedIndex = selectedIndex >= 0 ? selectedIndex : null;
    renderRolloutList();
    if (selectFirst && state.records.length && state.selectedIndex === null) {
      await selectLine(state.records[0].line, 0);
    } else if (!state.records.length) {
      state.detailRequest += 1;
      state.selectedLine = null;
      state.selectedIndex = null;
      state.record = null;
      $("chat").innerHTML = `<div class="empty-state">No conversation selected.</div>`;
      $("overview").innerHTML = `<div class="empty-state">No outcome selected.</div>`;
      $("conversation-summary").textContent = "Select a rollout";
      updateIndexCount();
    }
  } catch (error) {
    if (request !== state.recordsRequest) return;
    $("rollout-list").innerHTML = `<div class="error-state">${esc(error.message)}</div>`;
  }
}

function parseJsonString(value) {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text || !/^[{[]/.test(text)) return value;
  try {
    return JSON.parse(text);
  } catch (_) {
    return value;
  }
}

function normalizeToolCall(raw, index) {
  const parsed = parseJsonString(raw);
  const call = asObject(parsed);
  const functionValue = parseJsonString(call.function);
  const functionCall = asObject(functionValue);
  const name = firstPresent(call.name, functionCall.name, `Tool call ${index + 1}`);
  let argumentsValue = firstPresent(call.arguments, functionCall.arguments, call.input, parsed);
  argumentsValue = parseJsonString(argumentsValue);
  return {
    name: String(name),
    arguments: typeof argumentsValue === "string" ? argumentsValue : JSON.stringify(argumentsValue, null, 2),
    raw: raw,
  };
}

function renderToolCalls(rawCalls) {
  const calls = asArray(rawCalls).map(normalizeToolCall);
  if (!calls.length) return "";
  const names = calls.map((call) => call.name).filter(Boolean);
  const noun = calls.length === 1 ? "tool call" : "tool calls";
  const nameSummary = names.length ? ` (${names.join(", ")})` : "";
  return `<details class="tool-block">
    <summary>🔧 ${calls.length} ${noun}${esc(nameSummary)}</summary>
    <div class="tool-content">${calls.map((call) => `<div class="tool-call">
      <div class="tool-name">${esc(call.name)}</div>
      <pre>${esc(call.arguments || "—")}</pre>
    </div>`).join("")}</div>
  </details>`;
}

function wordCount(value) {
  const text = String(value ?? "").trim();
  return text ? text.split(/\s+/).length : 0;
}

function renderTurn(rawMessage, index, source) {
  const message = asObject(rawMessage);
  const role = String(message.role || "unknown").toLowerCase();
  const roleClass = ["user", "assistant", "system", "tool"].includes(role) ? role : "unknown";
  const reasoning = toText(message.reasoning_content);
  const content = toText(message.content);
  const toolCalls = asArray(message.tool_calls);
  const collapsed = role === "system" ? " collapsed" : "";
  const turnId = `turn-${index}`;
  const copyParts = [`${role.toUpperCase()} (${source})`];
  if (reasoning) copyParts.push(`Reasoning:\n${reasoning}`);
  if (content) copyParts.push(content);
  if (toolCalls.length) copyParts.push(`Tool calls:\n${JSON.stringify(toolCalls, null, 2)}`);
  state.turnCopies.set(turnId, copyParts.join("\n\n"));
  const reasoningHtml = role === "assistant" && reasoning ? `<details class="reasoning-block" open>
    <summary>🧠 Reasoning (${wordCount(reasoning).toLocaleString()} words)</summary>
    <div class="reasoning-content markdown">${renderMarkdown(reasoning)}</div>
  </details>` : "";
  const answerLabel = role === "assistant" && reasoning ? `<div class="answer-label">Visible answer</div>` : "";
  const visibleContent = content ? `<div class="markdown">${renderMarkdown(content)}</div>` : `<div class="em-dash">—</div>`;
  const toolsHtml = renderToolCalls(toolCalls);
  return `<article id="${turnId}" class="turn-card role-${roleClass}${collapsed}">
    <div class="turn-head">
      <span class="role-label">${esc(role)}</span>
      <span class="turn-source">${esc(source)}</span>
      <span class="turn-spacer"></span>
      <button class="icon-button" type="button" data-action="copy-turn" data-turn="${turnId}" aria-label="Copy turn" title="Copy turn">⧉</button>
      <button class="icon-button" type="button" data-action="toggle-turn" data-turn="${turnId}" aria-label="Collapse or expand turn" aria-expanded="${collapsed ? "false" : "true"}"><span class="chevron">⌄</span></button>
    </div>
    <div class="turn-body">${reasoningHtml}${answerLabel}${visibleContent}${toolsHtml}</div>
  </article>`;
}

function renderConversation(record) {
  const rollout = asObject(record.rollout);
  const prompt = asArray(rollout.prompt);
  const completion = asArray(rollout.completion);
  const messages = [
    ...prompt.map((message) => ({message, source: "prompt"})),
    ...completion.map((message) => ({message, source: "completion"})),
  ];
  state.turnCopies.clear();
  $("conversation-summary").textContent = `${messages.length.toLocaleString()} message${messages.length === 1 ? "" : "s"} · ${completion.length.toLocaleString()} completion turn${completion.length === 1 ? "" : "s"}`;
  $("chat").innerHTML = messages.length
    ? messages.map((item, index) => renderTurn(item.message, index, item.source)).join("")
    : `<div class="empty-state">This rollout has no recorded prompt or completion messages.</div>`;
  $("chat").scrollTop = 0;
}

function metricValue(record, key) {
  const rollout = asObject(record.rollout);
  const rewardCalculation = asObject(record.reward_calculation);
  const containers = [
    rollout,
    asObject(rollout.metrics),
    asObject(rewardCalculation.metrics),
    rewardCalculation,
    record,
  ];
  for (const container of containers) {
    if (Object.prototype.hasOwnProperty.call(container, key) && isPresent(container[key])) return container[key];
  }
  return null;
}

function humanizeKey(value) {
  return String(value ?? "")
    .replace(/^anti_hacking_/, "")
    .replace(/[._]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function guardLabel(key) {
  const labels = {
    anti_hacking_multiplier: "Overall guard multiplier",
    anti_hacking_format_reward: "Format score",
    anti_hacking_format_reward_weight: "Format score weight",
    anti_hacking_coherency: "Answer coherency",
    anti_hacking_reasoning_ethics: "Reasoning ethics issue",
    anti_hacking_reasoning_coherency: "Reasoning coherency",
    anti_hacking_meta_commentary: "Meta-commentary issue",
    anti_hacking_reasoning_quality: "Reasoning quality",
    anti_hacking_reasoning_words: "Reasoning word count",
    anti_hacking_reasoning_traces: "Reasoning traces",
    anti_hacking_truncated: "Response truncated",
    anti_hacking_missing_reasoning: "Missing reasoning trace",
    anti_hacking_judge_failures: "Guard judge failures",
    anti_hacking_unclosed_think: "Unclosed reasoning block",
    anti_hacking_zero_visible_words: "No visible answer words",
    anti_hacking_local_meta: "Local meta-commentary issue",
    anti_hacking_renderer_stripped_tool_calls: "Renderer stripped tool calls",
    anti_hacking_renderer_tool_call_fallback: "Renderer tool-call fallback",
  };
  return labels[key] || humanizeKey(key);
}

function guardTooltip(key) {
  const tooltips = {
    anti_hacking_multiplier: "Overall guard penalty on the task reward (1.0 = no penalty; 0.1/0.01 = soft penalty; 0.0 = hard-zeroed).",
    anti_hacking_missing_reasoning: "Hard-zeroed when an assistant turn has no reasoning trace; this guard requires one.",
    anti_hacking_truncated: "Hard-zeroed when the response is cut off by a token or timeout limit.",
    anti_hacking_coherency: "A value of -1 means the guard judge did not return a usable verdict; it fails open and does not penalize the rollout.",
    anti_hacking_reasoning_ethics: "A value of -1 means the guard judge did not return a usable verdict; it fails open and does not penalize the rollout.",
    anti_hacking_reasoning_coherency: "A value of -1 means the guard judge did not return a usable verdict; it fails open and does not penalize the rollout.",
    anti_hacking_meta_commentary: "A value of -1 means the guard judge did not return a usable verdict; it fails open and does not penalize the rollout.",
  };
  return tooltips[key] || "An anti-hacking guard signal recorded while scoring this rollout.";
}

const POSITIVE_GUARD_METRICS = new Set([
  "anti_hacking_coherency",
  "anti_hacking_reasoning_coherency",
  "anti_hacking_reasoning_quality",
  "anti_hacking_format_reward",
  "anti_hacking_reasoning_traces",
]);
const FLAG_GUARD_METRICS = new Set([
  "anti_hacking_reasoning_ethics",
  "anti_hacking_meta_commentary",
  "anti_hacking_local_meta",
  "anti_hacking_missing_reasoning",
  "anti_hacking_truncated",
  "anti_hacking_unclosed_think",
  "anti_hacking_zero_visible_words",
  "anti_hacking_renderer_stripped_tool_calls",
  "anti_hacking_renderer_tool_call_fallback",
  "anti_hacking_judge_failures",
]);

function metricDisplay(key, value, guard = false) {
  if (!isPresent(value)) return `<span class="em-dash">—</span>`;
  const number = Number(value);
  if (guard && Number.isFinite(number) && number === -1) {
    return `<span class="metric-pill neutral has-tooltip" tabindex="0" data-tooltip="The guard judge didn't return a usable verdict (fails open — not penalized).">No verdict</span>`;
  }
  if (key === "anti_hacking_multiplier" && Number.isFinite(number)) {
    if (number === 1) return `<span class="metric-pill good">No penalty · ${esc(fmtNumber(number))}</span>`;
    if (number === 0) return `<span class="metric-pill bad">Hard zero · 0</span>`;
    if (number < 1) return `<span class="metric-pill warn">Penalty · ${esc(fmtNumber(number))}</span>`;
  }
  if (Number.isFinite(number) && (number === 0 || number === 1)) {
    if (POSITIVE_GUARD_METRICS.has(key)) {
      const pass = number === 1;
      return `<span class="metric-pill ${pass ? "good" : "bad"}">${pass ? "Pass" : "Flag"} · ${number}</span>`;
    }
    if (FLAG_GUARD_METRICS.has(key)) {
      const pass = number === 0;
      return `<span class="metric-pill ${pass ? "good" : "bad"}">${pass ? "Pass" : "Flag"} · ${number}</span>`;
    }
    if (["mcqa_correct", "math_verify_score", "judge_score"].includes(key)) {
      const pass = number === 1;
      return `<span class="metric-pill ${pass ? "good" : "bad"}">${pass ? "Pass" : "Flag"} · ${number}</span>`;
    }
  }
  return esc(fmtValue(value));
}

function renderMetricRow(label, key, value, guard = false, tooltip = "") {
  const tooltipClass = tooltip ? " has-tooltip" : "";
  const tooltipAttributes = tooltip ? ` tabindex="0" data-tooltip="${esc(tooltip)}"` : "";
  return `<div class="metric-row">
    <span class="metric-name${tooltipClass}"${tooltipAttributes}>${esc(label)}</span>
    <span class="metric-value">${metricDisplay(key, value, guard)}</span>
  </div>`;
}

function findRecordedFormula(record) {
  const rollout = asObject(record.rollout);
  const calculation = asObject(record.reward_calculation);
  const direct = [
    asObject(rollout.metrics).final_reward_formula,
    asObject(rollout.anti_hacking_breakdown).final_reward_formula,
    calculation.final_reward_formula,
    asArray(calculation.final_reward_formulas)[0],
    asObject(calculation.metrics).final_reward_formula,
  ];
  const found = direct.find(isPresent);
  if (found) return String(found);
  for (const component of Object.values(asObject(calculation.components))) {
    const formula = asObject(component).final_reward_formula;
    if (isPresent(formula)) return String(formula);
  }
  return "";
}

function friendlyFormula(source) {
  return String(source)
    .replaceAll("anti_hacking_format_reward_weight", "format weight")
    .replaceAll("anti_hacking_format_reward", "format score")
    .replaceAll("anti_hacking_multiplier", "guard multiplier")
    .replaceAll("task_reward", "task reward")
    .replaceAll("final_reward", "final reward")
    .replaceAll("base_score", "base score")
    .replaceAll("*", "×")
    .replaceAll("_", " ");
}

function nestedFormulaValue(value, key, depth = 0) {
  if (depth > 6 || !value || typeof value !== "object" || Array.isArray(value)) return null;
  if (Object.prototype.hasOwnProperty.call(value, key) && isPresent(value[key]) && typeof value[key] !== "object") {
    return value[key];
  }
  for (const child of Object.values(value)) {
    const found = nestedFormulaValue(child, key, depth + 1);
    if (isPresent(found)) return found;
  }
  return null;
}

function formulaComponentValue(record, key) {
  const direct = metricValue(record, key);
  if (isPresent(direct)) return direct;
  const prefixed = metricValue(record, `decensor_${key}`);
  if (isPresent(prefixed)) return prefixed;
  return nestedFormulaValue(asObject(record.reward_calculation).components, key);
}

function pluggedCustomFormula(source, record) {
  let replacements = 0;
  const substituted = String(source).replace(/\b[A-Za-z_][A-Za-z0-9_]*\b/g, (key) => {
    const value = formulaComponentValue(record, key);
    if (!isPresent(value) || !Number.isFinite(Number(value))) return "—";
    replacements += 1;
    return fmtNumber(value, 4);
  });
  return replacements ? friendlyFormula(substituted) : "Recorded component values were not available.";
}

function formulaHtml(record) {
  const multiplier = metricValue(record, "anti_hacking_multiplier");
  const taskReward = metricValue(record, "task_reward");
  const formatReward = metricValue(record, "anti_hacking_format_reward");
  const weight = metricValue(record, "anti_hacking_format_reward_weight");
  const finalReward = firstPresent(metricValue(record, "final_reward"), metricValue(record, "reward"));
  const recorded = findRecordedFormula(record);
  const numbersAvailable = [multiplier, taskReward, formatReward, weight].every(
    (value) => isPresent(value) && Number.isFinite(Number(value)),
  );
  const taskOutsideBlend = /task_reward\s*\*\s*\(\(\s*1\s*-/.test(recorded);
  let explanation = "The task score and format score are combined, then the overall guard multiplier is applied.";
  let formula = "final reward = guard multiplier × ((1 − format weight) × task reward + format weight × format score)";
  let plugged = "Some formula inputs were not recorded for this rollout.";
  if (recorded && !/anti_hacking_multiplier|task_reward|final_reward/.test(recorded)) {
    explanation = "This environment recorded its own scoring rule.";
    formula = `Recorded rule: ${friendlyFormula(recorded)}`;
    plugged = /[+*()]/.test(recorded)
      ? `= ${pluggedCustomFormula(recorded, record)} = ${fmtValue(finalReward)}`
      : `Recorded final reward = ${fmtValue(finalReward)}`;
  } else if (taskOutsideBlend) {
    explanation = "The format adjustment scales the task score, then the overall guard multiplier is applied.";
    formula = "final reward = guard multiplier × task reward × ((1 − format weight) + format weight × format score)";
    if (numbersAvailable) {
      plugged = `= ${fmtNumber(multiplier, 3)} × ${fmtNumber(taskReward, 3)} × ((1 − ${fmtNumber(weight, 3)}) + ${fmtNumber(weight, 3)} × ${fmtNumber(formatReward, 3)}) = ${fmtNumber(finalReward, 4)}`;
    }
  } else if (numbersAvailable) {
    plugged = `= ${fmtNumber(multiplier, 3)} × ((1 − ${fmtNumber(weight, 3)}) × ${fmtNumber(taskReward, 3)} + ${fmtNumber(weight, 3)} × ${fmtNumber(formatReward, 3)}) = ${fmtNumber(finalReward, 4)}`;
  } else if (recorded) {
    formula = `Recorded rule: ${friendlyFormula(recorded)}`;
    plugged = `Recorded final reward = ${fmtValue(finalReward)}`;
  }
  const notes = [];
  const mathScore = metricValue(record, "math_verify_score");
  const mcqa = metricValue(record, "mcqa_correct");
  if (isPresent(mathScore)) {
    notes.push(`Math verification score = ${fmtValue(mathScore)} (${Number(mathScore) === 1 ? "symbolically correct" : "not verified as correct"}).`);
  }
  if (isPresent(mcqa)) {
    notes.push(`Multiple-choice result = ${fmtValue(mcqa)} (the selected answer was ${Number(mcqa) === 1 ? "correct" : "incorrect"}).`);
  }
  const recordedLine = recorded && friendlyFormula(recorded) !== formula.replace(/^Recorded rule: /, "")
    ? `<div class="formula-line">Recorded rule: ${esc(friendlyFormula(recorded))}</div>`
    : "";
  return `<div class="formula-card">
    <div class="formula-explanation">${esc(explanation)}</div>
    <div class="formula-line">${esc(formula)}</div>
    <div class="formula-line plugged">${esc(plugged)}</div>
    ${recordedLine}
    ${notes.length ? `<div class="verification-note">${notes.map(esc).join("<br>")}</div>` : ""}
  </div>`;
}

function statusChip(label, tone, tooltip) {
  return `<span class="status-chip ${tone} has-tooltip" tabindex="0" title="${esc(tooltip)}" data-tooltip="${esc(tooltip)}">${esc(label)}</span>`;
}

function statusHtml(record) {
  const labels = asObject(record.labels);
  const rollout = asObject(record.rollout);
  const filters = Object.keys(asObject(labels.filters)).length ? asObject(labels.filters) : asObject(rollout.filters);
  const chips = [];
  const dropReason = labels.drop_reason;
  if (dropReason === "post_batch_filter") {
    chips.push(statusChip("Dropped from training", "bad", "Excluded from the weight update after the batch was scored."));
  } else if (isPresent(dropReason)) {
    chips.push(statusChip(`Dropped: ${humanizeKey(dropReason)}`, "bad", "This rollout was excluded from the training update."));
  }
  const trainable = firstPresent(labels.is_trainable, rollout.is_trainable);
  if (trainable === false) chips.push(statusChip("Not trainable", "warn", "Not used for the gradient update this step."));
  if (trainable === true) chips.push(statusChip("Trainable", "good", "Eligible for the gradient update this step."));
  if (labels.in_train_batch === false) chips.push(statusChip("Not in this train batch", "neutral", "This rollout was not included in this step's training batch."));
  if (labels.in_train_batch === true) chips.push(statusChip("In this train batch", "good", "This rollout was included in this step's training batch."));
  const filtered = firstPresent(labels.is_filtered, rollout.is_filtered);
  if (filtered === true) chips.push(statusChip("Removed from batch", "bad", "Removed from the training batch."));
  if (filters.zero_advantage === true) {
    chips.push(statusChip("Zero advantage", "warn", "Every rollout in its group got the same reward, so it carries no learning signal (GRPO zero-advantage)."));
  }
  if (filters.gibberish === true) chips.push(statusChip("Gibberish flagged", "bad", "Flagged as garbled/gibberish output."));
  if (filters.repetition === true) chips.push(statusChip("Repetition flagged", "bad", "Flagged for a pathological repetition loop."));
  const stopCondition = firstPresent(labels.stop_condition, rollout.stop_condition);
  const stopLabels = {
    max_turns_reached: ["Ended: hit max turns", "warn", "Ended because it hit the max number of turns."],
    timeout_reached: ["Ended: timed out", "bad", "Ended because the rollout ran past its time limit."],
    reached_target_turns: ["Ended: target reached", "good", "Ended after completing the requested number of turns."],
    training_sample_too_long: ["Ended: sample too long", "bad", "The training sample exceeded the supported length."],
    malformed_assistant_turn_fail: ["Ended: malformed turn", "bad", "An assistant turn could not be parsed into the expected shape."],
  };
  if (isPresent(stopCondition)) {
    const status = stopLabels[stopCondition] || [`Ended: ${humanizeKey(stopCondition)}`, "neutral", "Recorded reason the rollout stopped."];
    chips.push(statusChip(...status));
  }
  const truncated = firstPresent(labels.is_truncated, rollout.is_truncated);
  if (truncated === true) chips.push(statusChip("Truncated", "bad", "The rollout response was cut off before normal completion."));
  if (Number(metricValue(record, "anti_hacking_missing_reasoning")) === 1) {
    chips.push(statusChip("Hard-zeroed: missing reasoning", "bad", "Hard-zeroed: an assistant turn had no reasoning trace (guard requires one)."));
  }
  if (Number(metricValue(record, "anti_hacking_truncated")) === 1) {
    chips.push(statusChip("Hard-zeroed: response cut off", "bad", "Hard-zeroed: the response was cut off (token/timeout limit)."));
  }
  if (isPresent(labels.error_type)) chips.push(statusChip(`Error: ${humanizeKey(labels.error_type)}`, "bad", "The rollout recorded an execution error."));
  return chips.length ? `<div class="status-list">${chips.join("")}</div>` : `<div class="status-list">${statusChip("No status flags", "neutral", "No filtering, truncation, or stop warning was recorded.")}</div>`;
}

function mergedJudgeLogs(record) {
  const logs = [...asArray(record.judge_logs), ...asArray(asObject(record.rollout).judge_logs)];
  const seen = new Set();
  return logs.filter((log) => {
    const item = asObject(log);
    const identity = JSON.stringify([item.kind, item.model, item.prompt, item.response, item.error]);
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  });
}

function judgeTitle(kindValue) {
  const kind = String(kindValue || "");
  const exact = {
    "nemotron_anti_hacking.coherency": "Guard: coherency",
    "nemotron_anti_hacking.reasoning_ethics": "Guard: reasoning ethics",
    "nemotron_anti_hacking.reasoning_coherency": "Guard: reasoning coherency",
    "nemotron_anti_hacking.meta_commentary": "Guard: meta-commentary",
    "decensor.coherency": "Guard: coherency",
    "decensor.reasoning_ethics": "Guard: reasoning ethics",
    "decensor.reasoning_coherency": "Guard: reasoning coherency",
    "decensor.meta_commentary": "Guard: meta-commentary",
    "decensor.refusal": "Task: refusal check",
    "decensor.md_presence": "Task: markdown present",
    "decensor.md_correctness": "Task: markdown correctness",
  };
  if (exact[kind]) return exact[kind];
  if (kind.endsWith(".equivalence")) return "Task: answer equivalence";
  if (kind.includes("calendar")) return "Task: instruction following";
  return humanizeKey(kind || "Judge");
}

function parsedVerdict(log) {
  const item = asObject(log);
  if (isPresent(item.error)) return {label: "Error", tone: "bad"};
  const response = toText(item.response);
  let score = null;
  const tagged = response.match(/<(?:score|refusal_score)>\s*(-?\d+(?:\.\d+)?)\s*<\/(?:score|refusal_score)>/i);
  if (tagged) score = Number(tagged[1]);
  if (!Number.isFinite(score) && isPresent(item.parsed_score) && Number.isFinite(Number(item.parsed_score))) {
    score = Number(item.parsed_score);
  }
  const kind = String(item.kind || "");
  if (kind === "decensor.refusal" && Number.isFinite(score)) {
    if (score <= 2) return {label: `Pass · ${fmtNumber(score, 3)}`, tone: "good"};
    if (score <= 5) return {label: `Caution · ${fmtNumber(score, 3)}`, tone: "warn"};
    return {label: `Flag · ${fmtNumber(score, 3)}`, tone: "bad"};
  }
  const flagWhenPositive = /(?:reasoning_ethics|meta_commentary|md_presence)$/.test(kind);
  if (Number.isFinite(score)) {
    if (score < 0) return {label: "No verdict", tone: "neutral"};
    const positive = score > 0;
    const pass = flagWhenPositive ? !positive : positive;
    return {label: `${pass ? "Pass" : "Flag"} · ${fmtNumber(score, 3)}`, tone: pass ? "good" : "bad"};
  }
  const labeled = response.match(/(?:verdict|answer|result)\s*[:=]\s*(yes|no)\b/i);
  const yesNo = labeled || Array.from(response.matchAll(/\b(yes|no)\b/gi)).at(-1);
  if (yesNo) {
    const yes = String(yesNo[1]).toLowerCase() === "yes";
    const pass = flagWhenPositive ? !yes : yes;
    return {label: pass ? "Pass" : "Flag", tone: pass ? "good" : "bad"};
  }
  return {label: "No verdict", tone: "neutral"};
}

function judgeLogsHtml(record) {
  const logs = mergedJudgeLogs(record);
  if (!logs.length) return `<div class="empty-state">No judge logs were recorded.</div>`;
  return logs.map((rawLog) => {
    const log = asObject(rawLog);
    const verdict = parsedVerdict(log);
    return `<div class="judge-card">
      <div class="judge-head">
        <div class="judge-title">${esc(judgeTitle(log.kind))}</div>
        <div class="judge-model">${esc(fmtValue(log.model))}</div>
        <span class="verdict-pill ${verdict.tone}">${esc(verdict.label)}</span>
      </div>
      ${isPresent(log.error) ? `<div class="judge-error">${esc(toText(log.error))}</div>` : ""}
      <details class="judge-raw">
        <summary>Prompt and response</summary>
        <div class="judge-raw-content">
          <div class="raw-label">Prompt</div><pre>${esc(toText(log.prompt) || "—")}</pre>
          <div class="raw-label">Response</div><pre>${esc(toText(log.response) || "—")}</pre>
        </div>
      </details>
    </div>`;
  }).join("");
}

function durationValue(value) {
  if (isPresent(value) && Number.isFinite(Number(value))) return Number(value);
  const item = asObject(value);
  if (Number.isFinite(Number(item.duration))) return Number(item.duration);
  if (Number.isFinite(Number(item.start)) && Number.isFinite(Number(item.end))) return Number(item.end) - Number(item.start);
  return null;
}

function totalTimingValue(timing, componentDurations) {
  const recorded = durationValue(timing.total);
  if (Number.isFinite(recorded)) return recorded;
  const starts = [timing.start_time, ...[timing.setup, timing.generation, timing.scoring].map((item) => asObject(item).start)]
    .filter((value) => isPresent(value) && Number.isFinite(Number(value)))
    .map(Number);
  const ends = [timing.setup, timing.generation, timing.scoring]
    .map((item) => asObject(item).end)
    .filter((value) => isPresent(value) && Number.isFinite(Number(value)))
    .map(Number);
  if (starts.length && ends.length) return Math.max(...ends) - Math.min(...starts);
  const known = componentDurations.filter((value) => Number.isFinite(value));
  return known.length ? known.reduce((total, value) => total + value, 0) : null;
}

function fmtDuration(value) {
  if (!isPresent(value) || !Number.isFinite(Number(value))) return "—";
  const seconds = Number(value);
  if (seconds < 0.001) return `${(seconds * 1000000).toFixed(0)} µs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(1)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(1)}s`;
}

function timingAndTokensHtml(record) {
  const rollout = asObject(record.rollout);
  const timing = asObject(rollout.timing);
  const setup = durationValue(timing.setup);
  const generation = durationValue(timing.generation);
  const scoring = durationValue(timing.scoring);
  const total = totalTimingValue(timing, [setup, generation, scoring]);
  const tokens = asObject(rollout.token_usage);
  const inputTokens = firstPresent(tokens.final_input_tokens, tokens.input_tokens);
  const outputTokens = firstPresent(tokens.final_output_tokens, tokens.output_tokens);
  return `<div class="mini-grid">
    <div class="mini-card">
      <div class="mini-card-title">Timing</div>
      <div class="mini-row"><span>Total</span><b>${esc(fmtDuration(total))}</b></div>
      <div class="mini-row"><span>Generation</span><b>${esc(fmtDuration(generation))}</b></div>
      <div class="mini-row"><span>Scoring</span><b>${esc(fmtDuration(scoring))}</b></div>
    </div>
    <div class="mini-card">
      <div class="mini-card-title">Token usage</div>
      <div class="mini-row"><span>Input</span><b>${esc(fmtValue(inputTokens))}</b></div>
      <div class="mini-row"><span>Output</span><b>${esc(fmtValue(outputTokens))}</b></div>
      <div class="mini-row"><span>Total</span><b>${esc(Number.isFinite(Number(inputTokens)) && Number.isFinite(Number(outputTokens)) ? fmtValue(Number(inputTokens) + Number(outputTokens)) : "—")}</b></div>
    </div>
  </div>`;
}

function renderOverview(record) {
  const rollout = asObject(record.rollout);
  const reward = firstPresent(metricValue(record, "reward"), metricValue(record, "final_reward"));
  const taskReward = metricValue(record, "task_reward");
  const finalReward = metricValue(record, "final_reward");
  const advantage = metricValue(record, "advantage");
  const taskMetrics = [
    ["Multiple-choice answer", "mcqa_correct"],
    ["Task reward", "task_reward"],
    ["Math verification", "math_verify_score"],
    ["Task judge score", "judge_score"],
    ["Assistant turns", "num_turns"],
  ];
  const guardValues = new Map();
  for (const container of [asObject(rollout.metrics), rollout, asObject(record.reward_calculation).metrics]) {
    for (const [key, value] of Object.entries(asObject(container))) {
      if (key.startsWith("anti_hacking_") && (value === null || ["string", "number", "boolean"].includes(typeof value))) {
        if (!guardValues.has(key) || !isPresent(guardValues.get(key))) guardValues.set(key, value);
      }
    }
  }
  const guardOrder = [
    "anti_hacking_multiplier", "anti_hacking_format_reward", "anti_hacking_format_reward_weight",
    "anti_hacking_coherency", "anti_hacking_reasoning_coherency", "anti_hacking_reasoning_ethics",
    "anti_hacking_meta_commentary", "anti_hacking_reasoning_quality", "anti_hacking_reasoning_words",
    "anti_hacking_reasoning_traces", "anti_hacking_missing_reasoning", "anti_hacking_truncated",
    "anti_hacking_judge_failures",
  ];
  const guardKeys = [...guardValues.keys()].sort((left, right) => {
    const leftIndex = guardOrder.indexOf(left);
    const rightIndex = guardOrder.indexOf(right);
    if (leftIndex >= 0 || rightIndex >= 0) return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
    return left.localeCompare(right);
  });
  const labels = asObject(record.labels);
  const identifier = isPresent(labels.example_id) ? `#${labels.example_id}` : String(labels.rollout_id || "").slice(0, 10) || `line ${record.line}`;
  $("overview-summary").textContent = `${labels.env_name || rollout.env_name || "Unknown environment"} · ${identifier}`;
  const judges = mergedJudgeLogs(record);
  $("overview").innerHTML = `
    <section class="overview-section">
      <div class="reward-card">
        <div class="reward-label">Reward</div>
        <div class="reward-value ${rewardClass(reward)}">${esc(fmtValue(reward))}</div>
        <div class="reward-parts">
          <div class="reward-part"><span>Task reward</span><b>${esc(fmtValue(taskReward))}</b></div>
          <div class="reward-part"><span>Final reward</span><b>${esc(fmtValue(finalReward))}</b></div>
          <div class="reward-part"><span>Advantage</span><b>${esc(fmtValue(advantage))}</b></div>
        </div>
      </div>
    </section>
    <section class="overview-section">
      <h3 class="section-title">How this reward was computed</h3>
      ${formulaHtml(record)}
    </section>
    <section class="overview-section">
      <h3 class="section-title">Metrics</h3>
      <div class="metric-group">
        <div class="metric-group-title">Task</div>
        ${taskMetrics.map(([label, key]) => renderMetricRow(label, key, metricValue(record, key))).join("")}
      </div>
      <div class="metric-group">
        <div class="metric-group-title">Guard (anti-hacking)</div>
        ${guardKeys.length ? guardKeys.map((key) => renderMetricRow(guardLabel(key), key, guardValues.get(key), true, guardTooltip(key))).join("") : `<div class="metric-row"><span class="metric-name">No guard metrics recorded</span><span class="metric-value em-dash">—</span></div>`}
      </div>
    </section>
    <section class="overview-section">
      <h3 class="section-title">Status</h3>
      ${statusHtml(record)}
    </section>
    <section class="overview-section">
      <h3 class="section-title">Judge logs · ${judges.length}</h3>
      ${judgeLogsHtml(record)}
    </section>
    <section class="overview-section">
      <h3 class="section-title">Usage</h3>
      ${timingAndTokensHtml(record)}
    </section>
    <details class="raw-json">
      <summary>Raw JSON</summary>
      <pre data-raw-json>Open to load the complete record…</pre>
    </details>`;
  const rawJson = $("overview").querySelector(".raw-json");
  rawJson.addEventListener("toggle", () => {
    const pre = rawJson.querySelector("[data-raw-json]");
    if (!rawJson.open || pre.dataset.loaded === "true") return;
    pre.textContent = JSON.stringify(record, null, 2);
    pre.dataset.loaded = "true";
  });
  $("overview").scrollTop = 0;
}

async function selectLine(line, index) {
  const request = ++state.detailRequest;
  state.selectedLine = Number(line);
  state.selectedIndex = Number(index);
  renderRolloutList();
  $("chat").innerHTML = `<div class="loading-state">Loading conversation…</div>`;
  $("overview").innerHTML = `<div class="loading-state">Loading outcome…</div>`;
  try {
    const data = await fetchJson(`/api/record?line=${encodeURIComponent(line)}`);
    if (request !== state.detailRequest) return;
    state.record = data.record;
    renderConversation(data.record);
    renderOverview(data.record);
  } catch (error) {
    if (request !== state.detailRequest) return;
    $("chat").innerHTML = `<div class="error-state">${esc(error.message)}</div>`;
    $("overview").innerHTML = `<div class="error-state">${esc(error.message)}</div>`;
  }
}

function resetAndLoad() {
  state.detailRequest += 1;
  state.offset = 0;
  state.snapshotLine = null;
  state.selectedLine = null;
  state.selectedIndex = null;
  loadRecords();
}

$("rollout-list").addEventListener("click", (event) => {
  const row = event.target.closest("[data-line]");
  if (row) selectLine(Number(row.dataset.line), Number(row.dataset.index));
});

$("chat").addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const turn = $(button.dataset.turn);
  if (button.dataset.action === "copy-turn") {
    copyText(state.turnCopies.get(button.dataset.turn) || "");
  } else if (button.dataset.action === "toggle-turn" && turn) {
    const collapsed = turn.classList.toggle("collapsed");
    button.setAttribute("aria-expanded", String(!collapsed));
  }
});

$("collapse-all").addEventListener("click", () => {
  for (const turn of $("chat").querySelectorAll(".turn-card")) {
    turn.classList.add("collapsed");
    const button = turn.querySelector('[data-action="toggle-turn"]');
    if (button) button.setAttribute("aria-expanded", "false");
  }
});

$("expand-all").addEventListener("click", () => {
  for (const turn of $("chat").querySelectorAll(".turn-card")) {
    turn.classList.remove("collapsed");
    const button = turn.querySelector('[data-action="toggle-turn"]');
    if (button) button.setAttribute("aria-expanded", "true");
  }
});

let searchTimer = null;
$("q").addEventListener("input", () => {
  window.clearTimeout(searchTimer);
  searchTimer = window.setTimeout(resetAndLoad, 250);
});
for (const id of ["env", "kind", "trainable", "reward"]) $(id).addEventListener("change", resetAndLoad);
$("reset-filters").addEventListener("click", () => {
  for (const id of ["q", "env", "kind", "trainable", "reward"]) $(id).value = "";
  resetAndLoad();
});
$("refresh").addEventListener("click", async () => {
  try {
    state.snapshotLine = null;
    await loadStats();
    await loadRecords();
  } catch (error) {
    showToast(error.message);
  }
});
$("previous-page").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - PAGE_SIZE);
  state.selectedLine = null;
  state.selectedIndex = null;
  loadRecords();
});
$("next-page").addEventListener("click", () => {
  if (state.offset + state.records.length >= state.totalMatched) return;
  state.offset += PAGE_SIZE;
  state.selectedLine = null;
  state.selectedIndex = null;
  loadRecords();
});

async function start() {
  try {
    await loadStats();
    await loadRecords();
  } catch (error) {
    $("rollout-list").innerHTML = `<div class="error-state">${esc(error.message)}</div>`;
  }
}

start();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a JSONL rollout viewer.")
    parser.add_argument("--file", required=True, type=Path, help="Path to rollouts.jsonl")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", default=8787, type=int, help="Bind port")
    return parser.parse_args()


def get_path_value(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def record_env(record: dict[str, Any]) -> str | None:
    return (
        get_path_value(record, "labels.env_name")
        or record.get("env_name")
        or get_path_value(record, "rollout.env_name")
    )


def first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def record_reward(record: dict[str, Any]) -> Any:
    if record.get("record_type") != "rollout":
        return None
    return first_value(
        get_path_value(record, "rollout.reward"),
        get_path_value(record, "rollout.final_reward"),
        get_path_value(record, "reward_calculation.reward"),
    )


def record_judge_logs(record: dict[str, Any]) -> list[Any]:
    top_level = record.get("judge_logs")
    nested = get_path_value(record, "rollout.judge_logs")
    logs = list(top_level) if isinstance(top_level, list) else []
    if isinstance(nested, list):
        for log in nested:
            if log not in logs:
                logs.append(log)
    return logs


def record_preview(record: dict[str, Any]) -> str:
    rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
    prompt = rollout.get("prompt") if isinstance(rollout.get("prompt"), list) else []
    completion = rollout.get("completion") if isinstance(rollout.get("completion"), list) else []
    messages = [message for message in [*prompt, *completion] if isinstance(message, dict)]
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return " ".join(str(message["content"]).split())[:360]
    for message in messages:
        if message.get("reasoning_content"):
            return " ".join(str(message["reasoning_content"]).split())[:360]
    return "—"


def parse_json_record(raw_line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "record_type": "parse_error",
            "error": repr(exc),
            "raw_line": raw_line[:2000].decode("utf-8", errors="replace"),
        }
    if isinstance(value, dict):
        return value
    return {
        "record_type": "parse_error",
        "error": f"Expected a JSON object, found {type(value).__name__}",
        "raw_line": raw_line[:2000].decode("utf-8", errors="replace"),
    }


def summarize_record(line: int, record: dict[str, Any]) -> dict[str, Any]:
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    rollout = record.get("rollout") if isinstance(record.get("rollout"), dict) else {}
    record_type = str(record.get("record_type", "unknown"))
    env_name = record_env(record)
    preview = record_preview(record) if record_type == "rollout" else "—"
    title_parts = [
        record_type,
        str(record.get("kind") or ""),
        f"env={env_name}" if env_name else "",
        f"example={labels.get('example_id')}" if labels.get("example_id") is not None else "",
        f"stop={labels.get('stop_condition')}" if labels.get("stop_condition") else "",
    ]
    if record_type.endswith("summary"):
        counts = record.get("counts") if isinstance(record.get("counts"), dict) else {}
        title_parts.append(" ".join(f"{key}={value}" for key, value in list(counts.items())[:4]))
    elif rollout.get("error"):
        title_parts.append(f"error={labels.get('error_type') or 'yes'}")
    summary = {
        "line": line,
        "record_type": record_type,
        "kind": record.get("kind"),
        "step": record.get("step"),
        "env_name": env_name,
        "example_id": labels.get("example_id"),
        "rollout_id": labels.get("rollout_id"),
        "is_trainable": labels.get("is_trainable"),
        "is_filtered": labels.get("is_filtered"),
        "title": " ".join(part for part in title_parts if part),
        "preview": preview,
        "reward": record_reward(record),
        "drop_reason": labels.get("drop_reason"),
        "judge_logs_count": len(record_judge_logs(record)),
    }
    summary["_search"] = json.dumps(
        {
            "title": summary["title"],
            "preview": preview,
            "record_type": record_type,
            "kind": summary["kind"],
            "env_name": env_name,
            "step": summary["step"],
            "labels": labels,
            "reward_calculation": record.get("reward_calculation"),
        },
        default=str,
        ensure_ascii=False,
    ).lower()
    return summary


def reward_matches(value: Any, reward_filter: str) -> bool:
    if not reward_filter:
        return True
    if value is None:
        return reward_filter == "missing"
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return reward_filter == "missing"
    if reward_filter == "high":
        return reward >= 0.7
    if reward_filter == "mid":
        return 0.3 <= reward < 0.7
    if reward_filter == "low":
        return reward < 0.3
    return True


def summary_matches(summary: dict[str, Any], filters: dict[str, str]) -> bool:
    if filters.get("kind") and str(summary.get("kind", "")) != filters["kind"]:
        return False
    if filters.get("type") and str(summary.get("record_type", "")) != filters["type"]:
        return False
    if filters.get("env") and filters["env"].lower() not in str(summary.get("env_name") or "").lower():
        return False
    if filters.get("step") and str(summary.get("step", "")) != filters["step"]:
        return False
    if filters.get("trainable"):
        expected = filters["trainable"].lower() == "true"
        if summary.get("is_trainable") is not expected:
            return False
    if not reward_matches(summary.get("reward"), filters.get("reward", "")):
        return False
    return not filters.get("q") or filters["q"].lower() in str(summary.get("_search") or "")


class RolloutStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.identity: tuple[int, int] | None = None
        self.position = 0
        self.summaries: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.envs: Counter[str] = Counter()

    def _reset(self, identity: tuple[int, int]) -> None:
        self.identity = identity
        self.position = 0
        self.summaries.clear()
        self.counts.clear()
        self.envs.clear()

    def refresh(self) -> None:
        with self.lock:
            if not self.path.exists():
                self.path = resolve_jsonl_path(self.path)
                if not self.path.exists():
                    return
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.position:
                self._reset(identity)

            with self.path.open("rb") as handle:
                handle.seek(self.position)
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        self.position = handle.tell()
                        break
                    if not raw_line.endswith(b"\n"):
                        self.position = offset
                        break

                    line_no = len(self.summaries) + 1
                    record = parse_json_record(raw_line)
                    summary = summarize_record(line_no, record)
                    summary["_offset"] = offset
                    self.summaries.append(summary)
                    record_type = str(record.get("record_type", "unknown"))
                    self.counts[record_type] += 1
                    env_name = record_env(record)
                    if env_name:
                        self.envs[str(env_name)] += 1
                    self.position = handle.tell()

    def stats(self) -> dict[str, Any]:
        self.refresh()
        with self.lock:
            return {
                "file": str(self.path),
                "total_lines": len(self.summaries),
                "rollouts": self.counts.get("rollout", 0),
                "summaries": sum(value for key, value in self.counts.items() if key.endswith("summary")),
                "record_types": dict(self.counts),
                "envs": dict(self.envs),
            }

    def records(
        self,
        filters: dict[str, str],
        limit: int,
        offset: int,
        order: str,
        max_line: int | None,
    ) -> dict[str, Any]:
        self.refresh()
        rows: list[dict[str, Any]] = []
        matched = 0
        with self.lock:
            snapshot_line = min(max_line or len(self.summaries), len(self.summaries))
            snapshot = self.summaries[:snapshot_line]
            summaries = reversed(snapshot) if order == "desc" else iter(snapshot)
            for summary in summaries:
                if not summary_matches(summary, filters):
                    continue
                if matched >= offset and len(rows) < limit:
                    rows.append({key: value for key, value in summary.items() if not key.startswith("_")})
                matched += 1
            return {
                "total_lines": len(self.summaries),
                "total_matched": matched,
                "offset": offset,
                "limit": limit,
                "snapshot_line": snapshot_line,
                "has_more": offset + len(rows) < matched,
                "records": rows,
            }

    def record(self, line: int) -> dict[str, Any] | None:
        self.refresh()
        with self.lock:
            if line <= 0 or line > len(self.summaries):
                return None
            offset = int(self.summaries[line - 1]["_offset"])
            with self.path.open("rb") as handle:
                handle.seek(offset)
                record = parse_json_record(handle.readline())
            record["line"] = line
            if "judge_logs" not in record:
                record["judge_logs"] = record_judge_logs(record)
            return record


def parse_query_int(query: dict[str, list[str]], key: str, default: int) -> int:
    values = query.get(key)
    if not values:
        return default
    value = values[0].strip()
    if not value.lstrip("-").isdigit():
        raise ValueError(f"{key} must be an integer")
    return int(value)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RolloutViewer/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def handle(self) -> None:
        # Browsers and polling clients drop the socket mid-response (tab switch,
        # request superseded by the next poll); that surfaces as a broken pipe
        # during the write. It's expected, not a server fault — swallow it.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @property
    def rollout_store(self) -> RolloutStore:
        return self.server.rollout_store  # type: ignore[attr-defined]

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html()
            return
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/stats":
                self.send_json(self.rollout_store.stats())
            elif parsed.path == "/api/records":
                self.send_json(self.api_records(query))
            elif parsed.path == "/api/record":
                self.send_json(self.api_record(query))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def api_records(self, query: dict[str, list[str]]) -> dict[str, Any]:
        limit = max(1, min(parse_query_int(query, "limit", 250), 2000))
        offset = max(0, parse_query_int(query, "offset", 0))
        max_line_value = parse_query_int(query, "max_line", 0)
        max_line = max(1, max_line_value) if max_line_value else None
        order = query.get("order", ["desc"])[0]
        if order not in {"asc", "desc"}:
            raise ValueError("order must be 'asc' or 'desc'")
        ignored = {"limit", "offset", "max_line", "order"}
        filters = {key: values[0] for key, values in query.items() if values and key not in ignored}
        return self.rollout_store.records(filters, limit, offset, order, max_line)

    def api_record(self, query: dict[str, list[str]]) -> dict[str, Any]:
        line = parse_query_int(query, "line", 0)
        if line <= 0:
            return {"error": "line must be positive"}
        record = self.rollout_store.record(line)
        return {"record": record} if record is not None else {"error": f"line {line} not found"}


def resolve_jsonl_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() or resolved.name != "rollouts.jsonl":
        return resolved
    candidates = list(resolved.parent.parent.glob("*/rollouts/rollouts.jsonl"))
    return candidates[0].resolve() if len(candidates) == 1 else resolved


def main() -> int:
    args = parse_args()
    path = resolve_jsonl_path(args.file)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.rollout_store = RolloutStore(path)  # type: ignore[attr-defined]
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Serving {path} at http://{shown_host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
