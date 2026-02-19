export function routeTop1(hiddenState, routerWeights) {
  let bestExpert = 0;
  let bestLogit = Number.NEGATIVE_INFINITY;

  for (let expert = 0; expert < routerWeights.length; expert += 1) {
    const row = routerWeights[expert];
    let logit = 0;
    for (let i = 0; i < hiddenState.length; i += 1) {
      logit += hiddenState[i] * row[i];
    }

    if (logit > bestLogit) {
      bestLogit = logit;
      bestExpert = expert;
    }
  }

  return { expertId: bestExpert, logit: bestLogit };
}
