import type { AppState } from "./urlState";

/** Every station implements exactly this interface. */
export interface StationProps {
  state: AppState;
  update: (patch: Partial<AppState>) => void;
}
