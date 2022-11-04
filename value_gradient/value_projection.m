%% Plot policy against projection
clear all
syms q qd qdd real;
filename = "~/Desktop/cached_value.mat";
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

%% Compute values
for i = 1:length(poses)
    expr1 =  subs(v(1, i), q, poses(1, i));
    expr2 =  subs(pi, q, poses(1, i));
    for j = 1:length(vels)
        VALUES(i, j) = subs(expr1, qd, vels(1, j));
        CTRLS(i, j) = subs(expr2, qd, vels(1, j));
    end
end

[vq, vqd] = gradient(VALUES);
save(filename, 'VALUES');

%% Projection
close all
VALUES = load(filename).VALUES;

alpha = 50;
% Cost function and their derivatives
f = @(q, qd, qdd)([qd; qdd]);
v = @(q, qd)(sqrt(3) * q^2 + 2 * q * qd + sqrt(3) * qd^2);
fcost = @(q, qd, u)(q^2 + qd^2 + u^2);
v_jac = @(q, qd)([2*qd + 2*3^(1/2)*q; 2*q + 2*3^(1/2)*qd]);
v_jac_norm = @(q, qd)(sqrt(v_jac(q, qd)' * v_jac(q, qd) + 1e-6));

% Projection operator
proj_upper = @(q, qd, dfdt)(dfdt - v_jac(q, qd)/v_jac_norm(q, qd) * (v_jac(q, qd)' * dfdt + alpha * (q^2 + qd^2)));
proj_lower = @(q, qd, dfdt)(dfd - v_jac(q, qd) * v_jac(q, qd)' * dfdt / ...
    v_jac_norm(q, qd));

integrator = @(q, qd, qdd)([q + qd * 0.01; qd + qdd * 0.01]);

q_t = rand* 2; qd_t = rand * 2; qdd_t = 0;
figure;
hax1 = axes;
hold on;
contourf(POS, VEL, VALUES);
quiver(poses, vels, vq, vqd, 'b');
plot(q_t, qd_t, '-o');
title("J level sets");
xlabel("q");
ylabel("v");

for i = 1:50
    % compute the next state
    qh_t2 = q_t + qd_t * 0.01; qdh_t2 = qd_t + qdd_t * 0.01;
    plot(hax1, qh_t2, qdh_t2, '*');
    
    % assemble dxdt and project
    dfdt   = [qd_t; qdd_t];
    fnext  = proj_upper(q_t, qd_t, dfdt);
    
    % integrate to find next state along V
    q_hat_next = q_t + fnext(1) * 0.01;
    qd_hat_next = qd_t + fnext(2) * 0.01;
    
    % update state
    q_t = q_hat_next;
    qd_t = qd_hat_next;
    plot(hax1, q_hat_next, qd_hat_next, '+');
    drawnow;
end


