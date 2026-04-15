// Root App component — state provider, keyboard handler, tab switching, layout

import { useEffect } from "react";
import { useKeyboard } from "@opentui/react";
import { AppProvider, useAppState, useDispatch } from "./state/app-context";
import { TAB_LIST, type TabId } from "./state/types";
import { Header } from "./components/layout/header";
import { TabBar } from "./components/layout/tab-bar";
import { StatusBar } from "./components/layout/status-bar";
// import { TickerScroll } from "./components/layout/ticker-scroll";
import { MarketTab } from "./tabs/market";
import { PortfolioTab } from "./tabs/portfolio";
import { SignalsTab } from "./tabs/signals";
import { LookupTab } from "./tabs/lookup";
import { AgentsTab } from "./tabs/agents";
import { OrdersTab } from "./tabs/orders";
import { WatchlistTab } from "./tabs/watchlist";
import { KalimatiTab } from "./tabs/kalimati";
import { AccountTab } from "./tabs/account";
import { StrategiesTab } from "./tabs/strategies";
import { ModeSelectModal } from "./modals/mode-select";
import { TradeDialog } from "./modals/trade-dialog";
import { CommandPalette } from "./modals/command-palette";
import { LookupModal } from "./modals/lookup-modal";
import { BG_BASE } from "./theme/colors";
import { api } from "./data/api-client";
import { fmtTime } from "./components/ui/helpers";

// Tab key mapping
const KEY_TO_TAB: Record<string, TabId> = {};
for (const tab of TAB_LIST) {
  KEY_TO_TAB[tab.key] = tab.id;
}

function Dashboard() {
  const state = useAppState();
  const dispatch = useDispatch();

  // Check API connectivity
  useEffect(() => {
    const check = async () => {
      const ok = await api.ping();
      dispatch({ type: "SET_CONNECTED", connected: ok });
    };
    check();
    const timer = setInterval(check, 10_000);
    return () => clearInterval(timer);
  }, [dispatch]);

  // Global keyboard handler
  useKeyboard(
    (key) => {
      // Tab switching: keys 1-0
      if (!key.ctrl && !key.meta && !key.option) {
        const tab = KEY_TO_TAB[key.name];
        if (tab) {
          dispatch({ type: "SET_TAB", tab });
          return;
        }

        // Chart timeframes
        if (state.activeTab === "lookup") {
          const tfMap: Record<string, "D" | "W" | "M" | "Y" | "I"> = {
            d: "D",
            w: "W",
            m: "M",
            y: "Y",
            i: "I",
          };
          const tf = tfMap[key.name];
          if (tf) {
            dispatch({ type: "SET_CHART_TIMEFRAME", tf });
            return;
          }
        }

        // Actions
        switch (key.name) {
          case "b":
            dispatch({ type: "OPEN_MODAL", modal: "buy" });
            break;
          case "s":
            dispatch({ type: "OPEN_MODAL", modal: "sell" });
            break;
          case "l":
            dispatch({ type: "OPEN_MODAL", modal: "lookup" });
            break;
          case "a":
            dispatch({ type: "SET_TAB", tab: "agents" });
            break;
          case "r":
            dispatch({ type: "SET_LAST_REFRESH", time: fmtTime() });
            break;
          case "q":
            process.exit(0);
            break;
        }
      }

      // Ctrl+P or / for command palette
      if ((key.ctrl && key.name === "p") || (!key.ctrl && key.name === "/")) {
        dispatch({ type: "OPEN_MODAL", modal: "command_palette" });
      }
    },
    { release: false }
  );

  const closeModal = () => dispatch({ type: "CLOSE_MODAL" });

  // Render active tab
  const renderTab = () => {
    switch (state.activeTab) {
      case "market":
        return <MarketTab />;
      case "portfolio":
        return <PortfolioTab />;
      case "signals":
        return <SignalsTab />;
      case "lookup":
        return <LookupTab />;
      case "agents":
        return <AgentsTab />;
      case "orders":
        return <OrdersTab />;
      case "watchlist":
        return <WatchlistTab />;
      case "kalimati":
        return <KalimatiTab />;
      case "account":
        return <AccountTab />;
      case "strategies":
        return <StrategiesTab />;
      default:
        return null;
    }
  };

  // Render modal overlay
  const renderModal = () => {
    switch (state.modalOpen) {
      case "buy":
        return <TradeDialog side="BUY" onClose={closeModal} />;
      case "sell":
        return <TradeDialog side="SELL" onClose={closeModal} />;
      case "lookup":
        return <LookupModal onClose={closeModal} />;
      case "command_palette":
        return <CommandPalette onClose={closeModal} />;
      case "mode_select":
        return (
          <ModeSelectModal
            onSelect={(mode) => {
              dispatch({ type: "SET_MODE", mode });
              closeModal();
            }}
          />
        );
      default:
        return null;
    }
  };

  return (
    <box
      flexDirection="column"
      width="100%"
      height="100%"
      backgroundColor={BG_BASE}
    >
      <Header />
      <TabBar />
      {renderTab()}
      <StatusBar />
      {renderModal()}
    </box>
  );
}

export function App() {
  return (
    <AppProvider>
      <Dashboard />
    </AppProvider>
  );
}
