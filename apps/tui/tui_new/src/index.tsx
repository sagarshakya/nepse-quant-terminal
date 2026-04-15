import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";
import { App } from "./app";

async function main() {
  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    backgroundColor: "#05070a",
    enableMouseMovement: true,
  });

  createRoot(renderer).render(<App />);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
