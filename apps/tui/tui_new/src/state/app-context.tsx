import { createContext, useContext, useReducer, type ReactNode } from "react";
import { type AppState, type AppAction, initialState } from "./types";

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_TAB":
      return { ...state, activeTab: action.tab };
    case "SET_MODE":
      return { ...state, tradingMode: action.mode };
    case "SET_CONNECTED":
      return { ...state, connected: action.connected };
    case "SET_LAST_REFRESH":
      return { ...state, lastRefresh: action.time };
    case "SET_LOOKUP_SYMBOL":
      return { ...state, lookupSymbol: action.symbol };
    case "SET_CHART_TIMEFRAME":
      return { ...state, chartTimeframe: action.tf };
    case "OPEN_MODAL":
      return { ...state, modalOpen: action.modal };
    case "CLOSE_MODAL":
      return { ...state, modalOpen: null };
    default:
      return state;
  }
}

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppState(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppState must be used within AppProvider");
  return ctx.state;
}

export function useDispatch(): React.Dispatch<AppAction> {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useDispatch must be used within AppProvider");
  return ctx.dispatch;
}
