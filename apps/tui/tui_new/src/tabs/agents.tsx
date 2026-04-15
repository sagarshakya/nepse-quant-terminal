// Tab 5: Agents — AI analysis picks, focus detail, and chat interface

import { useState, useRef, useCallback } from "react";
import { useTerminalDimensions, useKeyboard } from "@opentui/react";
import { TextAttributes } from "@opentui/core";
import { useAgentAnalysis } from "../data/hooks";
import { api } from "../data/api-client";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { truncate, verdictColor } from "../components/ui/helpers";

// ── Picks table columns ────────────────────────────────────────────

const picksColumns: Column[] = [
  { id: "symbol", label: "Symbol", width: 10 },
  { id: "decision", label: "Verdict", width: 8 },
  { id: "score", label: "Score", width: 6, align: "right" },
  { id: "confidence", label: "Conf", width: 6, align: "right" },
  { id: "reasoning", label: "Reasoning", width: 28 },
];

// ── Chat message type ──────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

// ── Focus zones ────────────────────────────────────────────────────

type FocusZone = "picks" | "detail" | "chat";
const ZONES: FocusZone[] = ["picks", "detail", "chat"];

export function AgentsTab() {
  const { width: termW, height: termH } = useTerminalDimensions();
  const { data, loading, error, refresh } = useAgentAnalysis();

  const [focusIdx, setFocusIdx] = useState(0);
  const [pickIdx, setPickIdx] = useState(0);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [detailScroll, setDetailScroll] = useState(0);

  const focusZone = ZONES[focusIdx];

  // Send chat message
  const sendChat = useCallback(async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;

    const userMsg: ChatMessage = { role: "user", text: msg, timestamp: new Date() };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    setChatLoading(true);

    try {
      const { reply } = await api.agentChat(msg);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        text: reply,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errMsg: ChatMessage = {
        role: "assistant",
        text: `Error: ${err.message || "Failed to get response"}`,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, errMsg]);
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, chatLoading]);

  // Keyboard navigation
  useKeyboard(
    (key) => {
      if (key.name === "Tab" && !key.shift) {
        setFocusIdx((prev) => (prev + 1) % ZONES.length);
      } else if (key.name === "Tab" && key.shift) {
        setFocusIdx((prev) => (prev - 1 + ZONES.length) % ZONES.length);
      }

      // Refresh analysis
      if (key.name === "r" && focusZone === "picks") {
        refresh();
      }

      // Detail scroll
      if (focusZone === "detail") {
        if (key.name === "ArrowDown" || key.name === "j") {
          setDetailScroll((s) => s + 1);
        } else if (key.name === "ArrowUp" || key.name === "k") {
          setDetailScroll((s) => Math.max(0, s - 1));
        }
      }

      // Chat input
      if (focusZone === "chat") {
        if (key.name === "Return") {
          sendChat();
        }
      }
    },
    { release: false },
  );

  // ── Loading state ──────────────────────────────────────────────

  if (!data) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>
          {loading ? "Loading agent analysis..." : `Error: ${error}`}
        </text>
      </box>
    );
  }

  const picks = data.picks;
  const selectedPick = picks[pickIdx] || null;

  const availH = termH - 4;
  const leftW = Math.floor(termW * 0.45);
  const rightW = termW - leftW;

  // ── Detail content for selected pick ───────────────────────────

  const detailLines: { text: string; fg: string }[] = [];
  if (selectedPick) {
    detailLines.push({ text: `Symbol: ${selectedPick.symbol}`, fg: colors.FG_AMBER });
    detailLines.push({
      text: `Verdict: ${selectedPick.decision}`,
      fg: verdictColor(selectedPick.decision),
    });
    detailLines.push({ text: `Score: ${selectedPick.score.toFixed(1)}`, fg: colors.FG_PRIMARY });
    detailLines.push({
      text: `Confidence: ${(selectedPick.confidence * 100).toFixed(0)}%`,
      fg: colors.FG_PRIMARY,
    });
    detailLines.push({ text: "", fg: colors.FG_DIM });
    detailLines.push({ text: "REASONING:", fg: colors.FG_AMBER });

    // Word-wrap reasoning to fit panel width
    const maxLineW = rightW - 6;
    const reasonWords = selectedPick.reasoning.split(" ");
    let currentLine = "";
    for (const word of reasonWords) {
      if ((currentLine + " " + word).length > maxLineW && currentLine.length > 0) {
        detailLines.push({ text: currentLine, fg: colors.FG_PRIMARY });
        currentLine = word;
      } else {
        currentLine = currentLine ? currentLine + " " + word : word;
      }
    }
    if (currentLine) detailLines.push({ text: currentLine, fg: colors.FG_PRIMARY });

    if (selectedPick.red_flags.length > 0) {
      detailLines.push({ text: "", fg: colors.FG_DIM });
      detailLines.push({ text: "RED FLAGS:", fg: colors.LOSS_HI });
      for (const flag of selectedPick.red_flags) {
        detailLines.push({ text: `  \u2022 ${flag}`, fg: colors.LOSS });
      }
    }

    if (selectedPick.catalysts.length > 0) {
      detailLines.push({ text: "", fg: colors.FG_DIM });
      detailLines.push({ text: "CATALYSTS:", fg: colors.GAIN_HI });
      for (const cat of selectedPick.catalysts) {
        detailLines.push({ text: `  \u2022 ${cat}`, fg: colors.GAIN });
      }
    }
  }

  const visibleDetailLines = detailLines.slice(detailScroll, detailScroll + 15);
  const detailPanelH = Math.floor(availH * 0.5);
  const chatPanelH = availH - detailPanelH;

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* ── Status Bar ──────────────────────────────────────── */}
      <box
        height={1}
        backgroundColor={colors.BG_HEADER}
        paddingLeft={2}
        flexDirection="row"
        alignItems="center"
      >
        <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
          AGENT
        </text>
        <text fg={colors.FG_DIM}>{"  |  "}</text>
        <text fg={colors.FG_SECONDARY}>Regime: </text>
        <text fg={data.regime === "bull" ? colors.GAIN_HI : data.regime === "bear" ? colors.LOSS_HI : colors.YELLOW}>
          {data.regime.toUpperCase()}
        </text>
        <text fg={colors.FG_DIM}>{"  |  "}</text>
        <text fg={colors.FG_SECONDARY}>Updated: </text>
        <text fg={colors.FG_PRIMARY}>{data.timestamp}</text>
        <text fg={colors.FG_DIM}>{"  |  "}</text>
        <text fg={colors.FG_DIM}>r=refresh  Tab=focus</text>
      </box>

      {/* ── Main Content ────────────────────────────────────── */}
      <box flexDirection="row" flexGrow={1}>
        {/* ── Left: Picks Table ──────────────────────────────── */}
        <box flexDirection="column" width={leftW}>
          <BloombergPanel
            title="TOP 10 PICKS"
            subtitle={`${picks.length} picks`}
            focused={focusZone === "picks"}
            flexGrow={1}
          >
            <DataTable
              columns={picksColumns}
              data={picks}
              selectedIndex={pickIdx}
              onSelect={(idx) => {
                setPickIdx(idx);
                setDetailScroll(0);
              }}
              height={availH - 3}
              focused={focusZone === "picks"}
              emptyText="No agent picks available"
              renderCell={(pick, colId) => {
                switch (colId) {
                  case "symbol":
                    return {
                      text: pick.symbol,
                      style: { fg: colors.FG_AMBER, bold: true },
                    };
                  case "decision":
                    return {
                      text: pick.decision,
                      style: { fg: verdictColor(pick.decision), bold: true },
                    };
                  case "score":
                    return {
                      text: pick.score.toFixed(1),
                      style: { fg: pick.score >= 7 ? colors.GAIN : pick.score >= 4 ? colors.YELLOW : colors.LOSS },
                    };
                  case "confidence":
                    return {
                      text: `${(pick.confidence * 100).toFixed(0)}%`,
                      style: { fg: pick.confidence >= 0.7 ? colors.GAIN : colors.FG_SECONDARY },
                    };
                  case "reasoning":
                    return {
                      text: truncate(pick.reasoning, 28),
                      style: { fg: colors.FG_SECONDARY },
                    };
                  default:
                    return { text: "" };
                }
              }}
            />
          </BloombergPanel>
        </box>

        {/* ── Right Column: Detail + Chat ────────────────────── */}
        <box flexDirection="column" width={rightW}>
          {/* Focus Detail Panel */}
          <BloombergPanel
            title={selectedPick ? `DETAIL: ${selectedPick.symbol}` : "DETAIL"}
            subtitle={selectedPick ? selectedPick.decision : "Select a pick"}
            focused={focusZone === "detail"}
            height={detailPanelH}
          >
            <box flexDirection="column" paddingLeft={1} flexGrow={1}>
              {selectedPick ? (
                visibleDetailLines.map((line, i) => (
                  <box key={i} height={1}>
                    <text fg={line.fg}>{line.text}</text>
                  </box>
                ))
              ) : (
                <box justifyContent="center" alignItems="center" flexGrow={1}>
                  <text fg={colors.FG_DIM}>Select a pick from the table</text>
                </box>
              )}
              {detailLines.length > 15 && (
                <box height={1}>
                  <text fg={colors.FG_DIM}>
                    [{detailScroll + 1}-{Math.min(detailScroll + 15, detailLines.length)}/{detailLines.length}] j/k to scroll
                  </text>
                </box>
              )}
            </box>
          </BloombergPanel>

          {/* Chat Panel */}
          <BloombergPanel
            title="CHAT"
            subtitle={chatLoading ? "thinking..." : "ask the agent"}
            focused={focusZone === "chat"}
            flexGrow={1}
          >
            <box flexDirection="column" flexGrow={1}>
              {/* Chat messages scroll area */}
              <scrollbox flexGrow={1}>
                {chatMessages.length === 0 ? (
                  <box paddingLeft={1} height={1}>
                    <text fg={colors.FG_DIM}>
                      Ask the agent about any stock or market condition...
                    </text>
                  </box>
                ) : (
                  chatMessages.map((msg, i) => (
                    <box key={i} paddingLeft={1} flexDirection="column">
                      <box height={1} flexDirection="row">
                        <text
                          fg={msg.role === "user" ? colors.CYAN : colors.GAIN_HI}
                          attributes={TextAttributes.BOLD}
                        >
                          {msg.role === "user" ? "YOU" : "AGT"}
                        </text>
                        <text fg={colors.FG_DIM}>
                          {"  "}
                          {msg.timestamp.toLocaleTimeString("en-US", { hour12: false })}
                        </text>
                      </box>
                      <box paddingLeft={2}>
                        <text fg={colors.FG_PRIMARY}>{msg.text}</text>
                      </box>
                    </box>
                  ))
                )}
              </scrollbox>

              {/* Typing indicator */}
              {chatLoading && (
                <box height={1} paddingLeft={1}>
                  <text fg={colors.PURPLE}>Agent is typing...</text>
                </box>
              )}

              {/* Input field */}
              <box
                height={1}
                backgroundColor={colors.BG_INPUT}
                paddingLeft={1}
                flexDirection="row"
              >
                <text fg={colors.FG_AMBER}>{"\u276f "}</text>
                <input
                  value={chatInput}
                  onChange={(val: string) => setChatInput(val)}
                  placeholder="Type a question..."
                  backgroundColor={colors.BG_INPUT}
                  textColor={colors.FG_BRIGHT}
                  focused={focusZone === "chat"}
                  flexGrow={1}
                />
              </box>
            </box>
          </BloombergPanel>
        </box>
      </box>
    </box>
  );
}
