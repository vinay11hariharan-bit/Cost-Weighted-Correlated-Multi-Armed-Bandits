%% BETA_THRESHOLD_NUMERIC
%  Numerical study of the stopping threshold beta(t,delta) for the
%  cost-weighted correlated two-arm bandit.
%
%  WHAT THIS COMPUTES
%    At each time t it evaluates
%
%        R_t^emp  =  sup_{Theta} L_t  /  q_t
%
%    where the numerator is the maximised likelihood over the compact
%    parameter set Theta = [-B,B]^2 x [-1+eps0, 1-eps0], and q_t is the
%    mixture density built from a uniform prior on the mean-box and a
%    uniform prior on a finite grid of correlations.  The threshold is
%    then  beta(t,delta) = log( R_t / delta ).
%
%  TWO SEPARATE STUDIES ARE RUN
%
%    PART 1 (benign).  Data simulated from the true environment.  This
%    measures the redundancy ACTUALLY INCURRED in normal operation.  It
%    is a LOWER BOUND on R_t and it is not a tight one, because the
%    datasets that drive the supremum essentially never occur when
%    simulating from a well-behaved environment.
%
%    PART 2 (adversarial).  Hand-built data of the form U = V = A, which
%    strains the unit-variance constraint.  Section 5 of the proof shows
%    the ratio grows like A^2 with t held FIXED, so the supremum over all
%    data is infinite and only the good-event restriction makes it
%    finite.  This part probes the regime the proof actually has to
%    survive.
%
%  READING THE OUTPUT
%    An exponent fitted to PART 1 alone says almost nothing about the
%    true exponent in R_t ~ C t^alpha.  Watch for non-monotonicity in the
%    running maximum: R_t is a supremum over a set that only grows with
%    t, so it CANNOT decrease.  If the empirical maximum decreases, the
%    estimator has not converged and the fitted slope is noise.
%
%  METHOD NOTES
%    - The means are profiled out EXACTLY.  For fixed rho' the
%      log-likelihood is exactly quadratic in (mu1,mu2), so the maximiser
%      solves a 2x2 linear system and the Gaussian integral over the box
%      is (2*pi/sqrt(det Q)) times a rectangle probability.
%    - The rectangle probability is computed by a one-dimensional
%      quadrature using the conditional-normal decomposition, so no
%      Statistics Toolbox is required.
%    - All likelihood arithmetic is done in logs with a max-subtraction,
%      otherwise everything underflows within a few dozen pulls.
%
%  No toolboxes required.
%
% -------------------------------------------------------------------------

clear; close all; clc;
rng(2024);

%% ------------------------- Settings -------------------------------------

% True environment
mu1True  = 1.0;      % mean of arm 1
mu2True  = 0.0;      % mean of arm 2
rhoTrue  = 0.5;      % correlation of a joint pull

% Parameter set Theta
B        = 3.0;      % means confined to [-B, B]
eps0     = 0.10;     % correlation confined to [-1+eps0, 1-eps0]

% Risk level
delta    = 0.05;

% Simulation
Tmax     = 80;       % horizon
nRep     = 150;      % number of independent runs
tGrid    = unique(round(logspace(log10(4), log10(Tmax), 22)));

% Adversarial sweep
Asweep   = [1 2 3 5 8 12];       % magnitude of the straining data
nSweep   = [2 3 5 8 12 20 30];   % number of joint pulls

% Numerical grids
nRhoSup  = 601;      % grid used to maximise over rho'
nRhoMix  = 201;      % grid used as the PRIOR support for the mixture
nQuad    = 400;      % nodes for the rectangle-probability quadrature

rhoLo = -1 + eps0;  rhoHi = 1 - eps0;
rhoSup = linspace(rhoLo, rhoHi, nRhoSup);
rhoMix = linspace(rhoLo, rhoHi, nRhoMix);

fprintf('Environment: mu1=%.2f  mu2=%.2f  rho=%.2f\n', mu1True, mu2True, rhoTrue);
fprintf('Theta: means in [-%.1f, %.1f], rho in [%.2f, %.2f]\n', B, B, rhoLo, rhoHi);
fprintf('delta = %.3f,  %d replicates,  horizon %d\n\n', delta, nRep, Tmax);

%% ================= PART 1: benign simulation ============================

nT        = numel(tGrid);
logR      = nan(nRep, nT);
Zstat     = nan(nRep, nT);

Sigma = [1, rhoTrue; rhoTrue, 1];
Lchol = chol(Sigma, 'lower');

fprintf('PART 1  benign simulation');
for rep = 1:nRep
    if mod(rep, 10) == 0, fprintf('.'); end

    X1 = []; X2 = []; U = []; V = [];

    for t = 1:Tmax
        % ---- sampling rule -------------------------------------------
        % Forced exploration on the joint action, which is the only
        % action that reveals the correlation.  Otherwise alternate the
        % two arms so that their counts stay level.
        N1 = numel(X1); N2 = numel(X2); N12 = numel(U);
        if N12 < sqrt(t)
            act = 3;
        elseif N1 <= N2
            act = 1;
        else
            act = 2;
        end

        % ---- draw the observation -------------------------------------
        switch act
            case 1
                X1(end+1,1) = mu1True + randn;                     %#ok<SAGROW>
            case 2
                X2(end+1,1) = mu2True + randn;                     %#ok<SAGROW>
            case 3
                z = Lchol * randn(2,1);
                U(end+1,1) = mu1True + z(1);                       %#ok<SAGROW>
                V(end+1,1) = mu2True + z(2);                       %#ok<SAGROW>
        end

        % ---- evaluate only at the requested times ---------------------
        k = find(tGrid == t, 1);
        if isempty(k) || numel(U) < 2, continue; end

        [lsup, lq, Zt] = evaluateAtTime(X1, X2, U, V, B, rhoSup, rhoMix, nQuad);
        logR(rep, k)  = lsup - lq;
        Zstat(rep, k) = Zt;
    end
end
fprintf(' done.\n');

logRmax  = max(logR, [], 1, 'omitnan');
logRmed  = median(logR, 1, 'omitnan');
logR90   = quantile_simple(logR, 0.90);
betaEmp  = logRmax + log(1/delta);

% Running maximum: R_t is a supremum over a growing set, so the TRUE R_t
% is non-decreasing.  Enforcing this exposes how noisy the raw estimate is.
logRrun  = cummax_nan(logRmax);
fin      = logRmax(isfinite(logRmax));
nViol    = sum(diff(fin) < 0);

Zmed = median(Zstat, 1, 'omitnan');
Z10  = quantile_simple(Zstat, 0.10);
Z90  = quantile_simple(Zstat, 0.90);

good = isfinite(logRmax) & tGrid > 5;
pFit = polyfit(log(tGrid(good)), logRmax(good), 1);
alphaBenign = pFit(1);

%% ================= PART 2: adversarial sweep ============================

fprintf('PART 2  adversarial sweep');
logRadv = nan(numel(nSweep), numel(Asweep));
for i = 1:numel(nSweep)
    fprintf('.');
    n = nSweep(i);
    for j = 1:numel(Asweep)
        A = Asweep(j);
        % Straining data: both coordinates equal and large.  The model can
        % only accommodate this by pushing rho to its boundary, where the
        % likelihood peak becomes needle-thin.
        U = A*ones(n,1);  V = A*ones(n,1);
        X1 = [];  X2 = [];
        [lsup, lq, ~] = evaluateAtTime(X1, X2, U, V, B, rhoSup, rhoMix, nQuad);
        logRadv(i,j) = lsup - lq;
    end
end
fprintf(' done.\n\n');

logRadvMax = max(logRadv, [], 2, 'omitnan')';
gA = isfinite(logRadvMax) & nSweep > 2;
if sum(gA) >= 2
    pA = polyfit(log(nSweep(gA)), logRadvMax(gA), 1);
    alphaAdv = pA(1);
else
    alphaAdv = NaN;
end

%% ------------------------- Report ---------------------------------------

fprintf('=== PART 1: benign data ===\n');
fprintf('%6s %12s %12s %12s %12s\n','t','log R (med)','log R (max)','beta','median Z_t');
fprintf('%s\n', repmat('-',1,60));
for k = 1:nT
    fprintf('%6d %12.3f %12.3f %12.3f %12.3f\n', ...
        tGrid(k), logRmed(k), logRmax(k), betaEmp(k), Zmed(k));
end
fprintf('\nFitted exponent on benign data: %.3f\n', alphaBenign);
fprintf('Non-monotone steps in the empirical maximum: %d\n', nViol);
if nViol > 0
    fprintf(['  --> R_t is a supremum over a set that only grows with t, so the\n' ...
             '      TRUE R_t cannot decrease.  These decreases are sampling noise\n' ...
             '      in an extreme order statistic.  The fitted slope inherits it.\n']);
end

fprintf('\n=== PART 2: adversarial data (U = V = A) ===\n');
fprintf('%6s', 'n');
for j = 1:numel(Asweep), fprintf('%10s', sprintf('A=%g', Asweep(j))); end
fprintf('\n%s\n', repmat('-', 1, 6 + 10*numel(Asweep)));
for i = 1:numel(nSweep)
    fprintf('%6d', nSweep(i));
    for j = 1:numel(Asweep)
        if isfinite(logRadv(i,j)), fprintf('%10.2f', logRadv(i,j));
        else,                      fprintf('%10s', '-'); end
    end
    fprintf('\n');
end
if isfinite(alphaAdv)
    fprintf('\nFitted exponent on adversarial data: %.3f\n', alphaAdv);
end

fprintf('\n--- How to read this ---\n');
fprintf('The benign maximum over %d runs is a LOWER bound on a supremum over\n', nRep);
fprintf('ALL datasets in the good event.  Section 5 of the proof shows the ratio\n');
fprintf('grows like A^2 with t held FIXED, so the unrestricted supremum is\n');
fprintf('infinite.  An exponent fitted to benign data alone says little about\n');
fprintf('the true exponent, and cannot be quoted as evidence that the proved\n');
fprintf('exponent of 2 is loose by a factor of t.\n');

%% ------------------------- Plots ----------------------------------------

figure('Position', [60 60 1220 860], 'Color', 'w');

% --- (a) the threshold itself ---
subplot(2,3,1); hold on; grid on; box on;
plot(tGrid, betaEmp, 'o-', 'LineWidth', 1.8, 'MarkerSize', 5, ...
     'Color', [0.10 0.35 0.75], 'MarkerFaceColor', [0.10 0.35 0.75]);
c15 = mean(betaEmp(good) - 1.5*log(tGrid(good)));
c20 = mean(betaEmp(good) - 2.0*log(tGrid(good)));
plot(tGrid, 1.5*log(tGrid) + c15, '--', 'LineWidth', 1.4, 'Color', [0.85 0.33 0.10]);
plot(tGrid, 2.0*log(tGrid) + c20, ':',  'LineWidth', 1.6, 'Color', [0.20 0.60 0.25]);
yline(log(1/delta), '-.', 'LineWidth', 1.2, 'Color', [0.5 0.5 0.5]);
xlabel('t  (total pulls)'); ylabel('\beta(t,\delta)');
title('(a)  Threshold vs time  (benign)');
legend({'empirical  log(R_t^{emp}/\delta)','1.5 log t + c','2.0 log t + c', ...
        'log(1/\delta) alone'}, 'Location','southeast','FontSize',7);
set(gca,'XScale','log');

% --- (b) exponent diagnostic, benign ---
subplot(2,3,2); hold on; grid on; box on;
plot(log(tGrid), logRmax, 'o', 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.10 0.35 0.75], 'Color', [0.10 0.35 0.75]);
plot(log(tGrid), logRrun, '-', 'LineWidth', 1.2, 'Color', [0.5 0.5 0.5]);
plot(log(tGrid), polyval(pFit, log(tGrid)), '-', 'LineWidth', 1.6, ...
     'Color',[0.85 0.33 0.10]);
xlabel('log t'); ylabel('log R_t^{emp}');
title(sprintf('(b)  Benign slope = %.2f  (%d decreases)', alphaBenign, nViol));
legend({'empirical max','running max','linear fit'},'Location','northwest','FontSize',7);

% --- (c) statistic against threshold ---
subplot(2,3,3); hold on; grid on; box on;
fill([tGrid fliplr(tGrid)], [Z10 fliplr(Z90)], [0.10 0.35 0.75], ...
     'FaceAlpha', 0.15, 'EdgeColor','none');
plot(tGrid, Zmed,    '-', 'LineWidth', 2.0, 'Color', [0.10 0.35 0.75]);
plot(tGrid, betaEmp, '-', 'LineWidth', 2.0, 'Color', [0.85 0.33 0.10]);
xlabel('t  (total pulls)'); ylabel('value');
title('(c)  Z_t linear, \beta logarithmic');
legend({'Z_t, 10-90 pct','Z_t median','\beta(t,\delta)'}, ...
       'Location','northwest','FontSize',7);

% --- (d) spread of the redundancy ---
subplot(2,3,4); hold on; grid on; box on;
plot(tGrid, logRmed, 's-', 'LineWidth', 1.4, 'Color', [0.4 0.4 0.4], ...
     'MarkerFaceColor',[0.4 0.4 0.4], 'MarkerSize',4);
plot(tGrid, logR90,  '^-', 'LineWidth', 1.4, 'Color', [0.45 0.30 0.70], ...
     'MarkerFaceColor',[0.45 0.30 0.70], 'MarkerSize',4);
plot(tGrid, logRmax, 'o-', 'LineWidth', 1.8, 'Color', [0.10 0.35 0.75], ...
     'MarkerFaceColor',[0.10 0.35 0.75], 'MarkerSize',4);
xlabel('t  (total pulls)'); ylabel('log R_t^{emp}');
title('(d)  Median vs tail: the gap is the point');
legend({'median','90th pct','maximum'},'Location','northwest','FontSize',7);
set(gca,'XScale','log');

% --- (e) adversarial: ratio vs A ---
subplot(2,3,5); hold on; grid on; box on;
cmap = lines(numel(nSweep));
for i = 1:numel(nSweep)
    if all(~isfinite(logRadv(i,:))), continue; end
    plot(Asweep, logRadv(i,:), 'o-', 'LineWidth', 1.4, 'MarkerSize', 4, ...
         'Color', cmap(i,:), 'MarkerFaceColor', cmap(i,:), ...
         'DisplayName', sprintf('n=%d', nSweep(i)));
end
xlabel('A   (magnitude of straining data)'); ylabel('log R^{emp}');
title('(e)  Adversarial data U = V = A');
legend('Location','northwest','FontSize',7);

% --- (f) benign vs adversarial ---
subplot(2,3,6); hold on; grid on; box on;
plot(log(tGrid), logRmax, 'o-', 'LineWidth', 1.6, 'MarkerSize', 4, ...
     'Color',[0.10 0.35 0.75], 'MarkerFaceColor',[0.10 0.35 0.75]);
gg = isfinite(logRadvMax);
plot(log(nSweep(gg)), logRadvMax(gg), 's-', 'LineWidth', 1.8, 'MarkerSize', 5, ...
     'Color',[0.85 0.10 0.10], 'MarkerFaceColor',[0.85 0.10 0.10]);
xlabel('log t   (or log n for the adversarial family)');
ylabel('log R^{emp}');
title('(f)  What simulation misses');
legend({'benign, max over runs','adversarial, worst A'}, ...
       'Location','northwest','FontSize',7);

sgtitle(sprintf(['Numerical threshold study    \\mu_1=%.1f, \\mu_2=%.1f, ' ...
                 '\\rho=%.1f,  B=%.1f, \\epsilon_0=%.2f, \\delta=%.2f'], ...
                 mu1True, mu2True, rhoTrue, B, eps0, delta), 'FontSize', 11);


%% ======================= Local functions ================================

function [logSupL, logQ, Zt] = evaluateAtTime(X1, X2, U, V, B, rhoSup, rhoMix, nQuad)
% Returns log sup_Theta L_t, log q_t, and the stopping statistic Z_t.

    % ---- unrestricted supremum over Theta ----------------------------
    lpSup = arrayfun(@(r) profileLogLik(r, X1, X2, U, V, B), rhoSup);
    logSupL = max(lpSup);

    % ---- mixture density q_t -----------------------------------------
    % For each prior grid point, the mean-integral over the box is
    %   exp(lp) * (2*pi/sqrt(det Q)) * P[ N(muhat, inv(Q)) in box ]
    % and the uniform prior density on the box contributes 1/(4B^2).
    K = numel(rhoMix);
    logTerm = nan(1, K);
    for i = 1:K
        [lp, muhat, Q] = profileLogLik(rhoMix(i), X1, X2, U, V, B);
        detQ  = Q(1,1)*Q(2,2) - Q(1,2)*Q(2,1);
        Sig   = [Q(2,2), -Q(1,2); -Q(2,1), Q(1,1)] / detQ;   % inv(Q)
        pBox  = rectProb(muhat, Sig, B, nQuad);
        pBox  = max(pBox, realmin);
        logTerm(i) = lp + log(2*pi) - 0.5*log(detQ) + log(pBox) - log(4*B^2);
    end
    logQ = logSumExp(logTerm) - log(K);

    % ---- supremum over the alternative set ----------------------------
    % On Alt the two means collapse to a common value m, and for fixed
    % rho' that common value has a closed form involving only 1+rho'.
    lpAlt = arrayfun(@(r) altLogLik(r, X1, X2, U, V, B), rhoSup);
    Zt = logSupL - max(lpAlt);
end


function [lp, muhat, Q] = profileLogLik(r, X1, X2, U, V, B)
% Profile log-likelihood at correlation r: means profiled out exactly,
% then clipped to the box [-B,B]^2.
    N1 = numel(X1); N2 = numel(X2); N12 = numel(U);
    om = 1 - r^2;

    a = N1  + N12/om;
    b = N2  + N12/om;
    d = N12*r/om;
    p = sum(X1) + (sum(U) - r*sum(V))/om;
    q = sum(X2) + (sum(V) - r*sum(U))/om;

    det2  = a*b - d^2;
    m1    = ( b*p + d*q ) / det2;
    m2    = ( a*q + d*p ) / det2;

    m1 = min(max(m1, -B), B);
    m2 = min(max(m2, -B), B);

    muhat = [m1; m2];
    Q     = [a, -d; -d, b];
    lp    = logLik(m1, m2, r, X1, X2, U, V);
end


function lp = altLogLik(r, X1, X2, U, V, B)
% Supremum of the log-likelihood at correlation r over the alternative
% set, where mu1 = mu2 = m.  Only 1+r appears in the equation for m.
    N1 = numel(X1); N2 = numel(X2); N12 = numel(U);
    w  = N12/(1+r);
    m  = ( sum(X1) + sum(X2) + (sum(U)+sum(V))/(1+r) ) / ( N1 + N2 + 2*w );
    m  = min(max(m, -B), B);
    lp = logLik(m, m, r, X1, X2, U, V);
end


function ll = logLik(m1, m2, r, X1, X2, U, V)
% Full log-likelihood, every term written out.
    N1 = numel(X1); N2 = numel(X2); N12 = numel(U);
    om = 1 - r^2;

    ll = -0.5*sum((X1 - m1).^2) - 0.5*sum((X2 - m2).^2) ...
         - 0.5*N1*log(2*pi) - 0.5*N2*log(2*pi);

    if N12 > 0
        aU = U - m1;
        bV = V - m2;
        quadForm = sum( aU.^2 - 2*r*(aU.*bV) + bV.^2 );
        ll = ll - N12*log(2*pi) - 0.5*N12*log(om) - quadForm/(2*om);
    end
end


function pr = rectProb(mu, Sig, B, nQuad)
% P[ Y in [-B,B]^2 ] for Y ~ N(mu, Sig), via the conditional-normal
% decomposition and a one-dimensional quadrature.  No toolbox.
    s1 = sqrt(Sig(1,1));
    if ~isfinite(s1) || s1 <= 0, pr = 1; return; end

    lo = max(-B, mu(1) - 8*s1);
    hi = min( B, mu(1) + 8*s1);
    if hi <= lo, pr = 0; return; end

    x  = linspace(lo, hi, nQuad);
    bt = Sig(2,1)/Sig(1,1);
    s2 = sqrt(max(Sig(2,2) - Sig(2,1)*bt, eps));

    condMean = mu(2) + bt*(x - mu(1));
    inner = normCdf((B - condMean)/s2) - normCdf((-B - condMean)/s2);
    dens  = exp(-0.5*((x - mu(1))/s1).^2) / (s1*sqrt(2*pi));

    pr = trapz(x, dens .* inner);
    pr = min(max(pr, 0), 1);
end


function y = normCdf(z)
    y = 0.5 * erfc(-z/sqrt(2));
end


function s = logSumExp(v)
    v = v(isfinite(v));
    if isempty(v), s = -Inf; return; end
    M = max(v);
    s = M + log(sum(exp(v - M)));
end


function q = quantile_simple(A, pLevel)
% Column-wise quantile without the Statistics Toolbox.
    [~, nc] = size(A);
    q = nan(1, nc);
    for j = 1:nc
        col = sort(A(isfinite(A(:,j)), j));
        if isempty(col), continue; end
        n = numel(col);
        idx = max(1, min(n, ceil(pLevel*n)));
        q(j) = col(idx);
    end
end


function y = cummax_nan(x)
% Running maximum, skipping NaN entries.
    y = x;  m = -Inf;
    for i = 1:numel(x)
        if isfinite(x(i)), m = max(m, x(i)); end
        y(i) = m;
    end
end