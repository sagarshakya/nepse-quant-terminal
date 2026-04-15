// Tab 9: Account — Paper account management

import { useState } from "react";
import { TextAttributes } from "@opentui/core";
import { useAccounts } from "../data/hooks";
import { api } from "../data/api-client";
import { BloombergPanel } from "../components/ui/panel";
import { DataTable, type Column, type CellStyle } from "../components/ui/data-table";
import * as colors from "../theme/colors";
import { fmtPrice, fmtNpr } from "../components/ui/helpers";
import type { PaperAccount } from "../data/types";

// ── Column definitions ──

const accountCols: Column[] = [
  { id: "name", label: "ACCOUNT", width: 14, align: "left" },
  { id: "nav", label: "NAV", width: 12, align: "right" },
  { id: "cash", label: "CASH", width: 12, align: "right" },
  { id: "status", label: "STATUS", width: 8, align: "center" },
  { id: "created_at", label: "CREATED", width: 12, align: "left" },
];

export function AccountTab() {
  const { data: accounts, loading, error, refresh } = useAccounts();
  const [focusedPanel, setFocusedPanel] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Form state
  const [newName, setNewName] = useState("");
  const [newCapital, setNewCapital] = useState("");
  const [formMessage, setFormMessage] = useState("");
  const [formFocusField, setFormFocusField] = useState<"name" | "capital">("name");

  const accountList = accounts ?? [];
  const activeAccount = accountList.find((a) => a.is_active);

  // Compute positions value from NAV - cash
  const positionsValue = activeAccount ? activeAccount.nav - activeAccount.cash : 0;
  const totalReturn = activeAccount && activeAccount.cash > 0
    ? ((activeAccount.nav / activeAccount.cash - 1) * 100)
    : 0;

  async function handleCreate() {
    const capital = parseFloat(newCapital);
    if (!newName.trim()) {
      setFormMessage("Name is required");
      return;
    }
    if (isNaN(capital) || capital <= 0) {
      setFormMessage("Invalid capital amount");
      return;
    }
    try {
      await api.createAccount(newName.trim(), capital);
      setFormMessage("Account created");
      setNewName("");
      setNewCapital("");
      refresh();
    } catch (err: any) {
      setFormMessage(`Error: ${err.message}`);
    }
  }

  async function handleActivate() {
    const account = accountList[selectedIndex];
    if (!account) return;
    try {
      await api.activateAccount(account.id);
      setFormMessage(`Activated: ${account.name}`);
      refresh();
    } catch (err: any) {
      setFormMessage(`Error: ${err.message}`);
    }
  }

  if (loading && !accounts) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.FG_DIM}>Loading accounts...</text>
      </box>
    );
  }

  if (error && !accounts) {
    return (
      <box flexGrow={1} justifyContent="center" alignItems="center">
        <text fg={colors.LOSS}>Error: {error}</text>
      </box>
    );
  }

  return (
    <box flexDirection="column" flexGrow={1} backgroundColor={colors.BG_BASE}>
      {/* Top: Active account summary */}
      <BloombergPanel title="ACTIVE ACCOUNT" subtitle={activeAccount?.name ?? "None"}>
        <box flexDirection="row" height={3} paddingLeft={1} paddingTop={1} gap={4}>
          {activeAccount ? (
            <>
              <box flexDirection="column">
                <text fg={colors.FG_SECONDARY}>Account</text>
                <text fg={colors.FG_AMBER} attributes={TextAttributes.BOLD}>
                  {activeAccount.name}
                </text>
              </box>
              <box flexDirection="column">
                <text fg={colors.FG_SECONDARY}>NAV</text>
                <text fg={colors.FG_PRIMARY} attributes={TextAttributes.BOLD}>
                  {fmtNpr(activeAccount.nav)}
                </text>
              </box>
              <box flexDirection="column">
                <text fg={colors.FG_SECONDARY}>Cash</text>
                <text fg={colors.CYAN}>{fmtNpr(activeAccount.cash)}</text>
              </box>
              <box flexDirection="column">
                <text fg={colors.FG_SECONDARY}>Positions</text>
                <text fg={colors.FG_PRIMARY}>{fmtNpr(positionsValue)}</text>
              </box>
              <box flexDirection="column">
                <text fg={colors.FG_SECONDARY}>Total Return</text>
                <text fg={colors.priceColor(totalReturn)}>
                  {totalReturn >= 0 ? "+" : ""}{totalReturn.toFixed(2)}%
                </text>
              </box>
            </>
          ) : (
            <text fg={colors.FG_DIM}>No active account. Create or activate one below.</text>
          )}
        </box>
      </BloombergPanel>

      {/* Bottom row: Account list | Create form */}
      <box flexDirection="row" flexGrow={1}>
        {/* Left: Account list */}
        <BloombergPanel
          title="ACCOUNTS"
          subtitle={`${accountList.length} total`}
          focused={focusedPanel === 0}
          flexGrow={2}
        >
          <DataTable
            columns={accountCols}
            data={accountList}
            height={18}
            focused={focusedPanel === 0}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            onActivate={handleActivate}
            emptyText="No accounts created"
            renderCell={(item: PaperAccount, colId) => {
              switch (colId) {
                case "name":
                  return {
                    text: item.name.slice(0, 14),
                    style: {
                      fg: item.is_active ? colors.FG_AMBER : colors.FG_PRIMARY,
                      bold: item.is_active,
                    },
                  };
                case "nav":
                  return { text: fmtPrice(item.nav), style: { fg: colors.FG_PRIMARY } };
                case "cash":
                  return { text: fmtPrice(item.cash), style: { fg: colors.CYAN } };
                case "status":
                  return {
                    text: item.is_active ? "ACTIVE" : "IDLE",
                    style: {
                      fg: item.is_active ? colors.GAIN : colors.FG_DIM,
                      bold: item.is_active,
                    },
                  };
                case "created_at":
                  return {
                    text: item.created_at.slice(0, 10),
                    style: { fg: colors.FG_SECONDARY },
                  };
                default:
                  return { text: "" };
              }
            }}
          />
          <box paddingLeft={1} height={1}>
            <text fg={colors.FG_DIM}>Enter: Activate selected  |  Tab: Switch panel</text>
          </box>
        </BloombergPanel>

        {/* Right: Create form */}
        <BloombergPanel
          title="CREATE ACCOUNT"
          focused={focusedPanel === 1}
          flexGrow={1}
        >
          <box flexDirection="column" paddingLeft={1} paddingTop={1} gap={1}>
            {/* Name field */}
            <box flexDirection="column">
              <text fg={colors.FG_SECONDARY}>Account Name</text>
              <input
                placeholder="e.g. Aggressive Growth"
                onInput={(v: string) => setNewName(v)}
                focused={focusedPanel === 1 && formFocusField === "name"}
              />
            </box>

            {/* Capital field */}
            <box flexDirection="column">
              <text fg={colors.FG_SECONDARY}>Starting Capital (NPR)</text>
              <input
                placeholder="e.g. 1000000"
                onInput={(v: string) => setNewCapital(v)}
                focused={focusedPanel === 1 && formFocusField === "capital"}
              />
            </box>

            {/* Create button */}
            <box height={1} paddingTop={1}>
              <box
                backgroundColor={colors.GAIN}
                paddingLeft={2}
                paddingRight={2}
              >
                <text fg={colors.BG_BASE} attributes={TextAttributes.BOLD}>
                  [ CREATE ]
                </text>
              </box>
            </box>

            {/* Status message */}
            {formMessage && (
              <box paddingTop={1}>
                <text
                  fg={
                    formMessage.startsWith("Error")
                      ? colors.LOSS
                      : colors.GAIN
                  }
                >
                  {formMessage}
                </text>
              </box>
            )}

            {/* Help text */}
            <box paddingTop={1}>
              <text fg={colors.FG_DIM}>Tab: Switch fields  |  Enter: Create</text>
            </box>
          </box>
        </BloombergPanel>
      </box>
    </box>
  );
}
