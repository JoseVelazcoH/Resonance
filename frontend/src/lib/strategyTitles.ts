export const STRATEGY_TITLES: Record<string, string> = {
  lift: "Lift Me Up",
  accompany: "Keep Me Company",
  energize: "Energize",
  calm: "Calm & Reflective",
};

export function strategyTitle(strategy: string): string {
  return STRATEGY_TITLES[strategy] ?? strategy;
}

export const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  lift: "Starts close to how you feel and gradually moves toward a brighter mood.",
  accompany: "Stays close to how you feel, matching your mood track after track.",
  energize: "Builds momentum with higher energy tracks to lift you up.",
  calm: "Eases the energy down to help you unwind and relax.",
};

export function strategyDescription(strategy: string): string {
  return STRATEGY_DESCRIPTIONS[strategy] ?? "A playlist shaped around your current mood.";
}
