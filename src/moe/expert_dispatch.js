import { routeTop1 } from "./router.js";

export function dispatchExpert(hiddenState, routerWeights, experts) {
  const { expertId } = routeTop1(hiddenState, routerWeights);
  const expert = experts[expertId];
  if (typeof expert !== "function") {
    throw new Error(`Missing expert function for id=${expertId}`);
  }

  return {
    expertId,
    output: expert(hiddenState),
  };
}
