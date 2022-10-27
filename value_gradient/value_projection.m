%% Plot policy against projection
clear all
syms q qd qdd real;
pi = -q + sqrt(3) * qd;
v  = sqrt(3) * q^2 + 2 * q * qd + sqrt(3) * qd^2;
vx = jacobian(v);
l  = q^2 + qd^2 + qdd^2;
f  = [qd; qdd];
dp = l + vx * f;
pi_d = diff(dp, qdd);
poses = -2:0.1:2;
vels = -2:0.1:2;
[POS, VEL] = meshgrid(poses, vels);
v = subs(v, q, poses);
VALUES = ones(size(POS));
CTRLS = ones(size(POS));

for i = 1:length(poses)
    expr1 =  subs(v(1, i), q, poses(1, i));
    expr2 =  subs(pi, q, poses(1, i));
    for j = 1:length(vels)
        VALUES(i, j) = subs(expr1, qd, vels(1, j));
        CTRLS(i, j) = subs(expr2, qd, vels(1, j));
    end
end

[vq, vqd] = gradient(VALUES);

%% Projection
close all

alpha = .00000001;
% Cost function and their derivatives
f = @(q, qd, qdd)([qd; qdd]);
v = @(q, qd)(sqrt(3) * q^2 + 2 * q * qd + sqrt(3) * qd^2);
fcost = @(q, qd, u)(q^2 + qd^2 + u^2);
v_jac = @(q, qd)([2*qd + 2*3^(1/2)*q; 2*q + 2*3^(1/2)*qd]);
v_jac_norm = @(q, qd)(sum(square(v_jac(q, qd))));

% Projection operator
proj_upper = @(q, qd, dfdt)(dfdt - v_jac(q, qd) * (v_jac(q, qd)' * dfdt + ...
    alpha * v(q, qd)) /v_jac_norm(q, qd));
proj_lower = @(q, qd, dfdt)(dfd - v_jac(q, qd) * v_jac(q, qd)' * dfdt / ...
    v_jac_norm(q, qd));

integrator = @(q, qd, qdd)([q + qd * 0.01; qd + qdd * 0.01]);

q = 1.2; qd = 1.2; qdd = 5;
q_next_h = q + qd * 0.01; qd_next_h = qd + qdd * 0.01;

hold on;
contourf(POS, VEL, VALUES);
quiver(poses, vels, vq, vqd, 'b');
plot(q, qd, '-o');
plot(q_next_h, qd_next_h, '*');

dfdt   = [qd_next_h; qdd];
fnext  = proj_upper(q, qd, dfdt);
state  = integrator(q, qd, fnext(2));
q_next = state(1); qd_next = state(2);

plot(q_next, qd_next, '+');
title("J level sets");
xlabel("q");
ylabel("v");

too_low  =  v_jac(q, qd)' * fnext <= -fcost(q, qd, 0);
too_high =  v_jac(q, qd)' * fnext + alpha * v(q, qd) >= 0;


