%% Plot policy against projection
clear all
syms q qd qdd real;
filename = "~/Desktop/cached_value.mat";
pi = -q + sqrt(3) * qd;
v  = sqrt(3) * q^2 + 2 * q * qd + sqrt(3) * qd^2;
vx = jacobian(v);
vqd = diff(v, qd);
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
[vq, vqd] = gradient(VALUES);

alpha = 30;


% Cost function and their derivatives
f = @(q, qd, qdd)([qd; qdd]);
v = @(q, qd)(sqrt(3) * q^2 + 2 * q * qd + sqrt(3) * qd^2);
fcost = @(q, qd, u)(q^2 + qd^2 + u^2);
v_jac = @(q, qd)([2*qd + 2*3^(1/2)*q; 2*q + 2*3^(1/2)*qd]);
v_qd = @(q, qd)(2*q - 2*3^(1/2)*(qd));
gauss_hessian = @(q, qd)(2 * (v_jac(q, qd)' * v_jac(q, qd) + 1e-6));
real_hessian  = [2*3^(1/2), 2; 2, 2*3^(1/2)];
sqr_norm = @(q, qd)(sqrt(v_jac(q, qd)' * v_jac(q, qd) + 1e-6));

% Projection operator
proj_upper = @(q, qd, dfdt)(dfdt - v_jac(q, qd)/sqr_norm(q, qd) * relu((v_jac(q, qd)' * dfdt + alpha * (q^2 + qd^2 + 0 * dfdt(2)^2))));
proj_lower = @(q, qd, dfdt)(dfd - v_jac(q, qd) * v_jac(q, qd)' * dfdt / ...
    v_jac_norm(q, qd));

integrator = @(q, qd, qdd)([q + qd * 0.01; qd + qdd * 0.01]);

% init =[0.738493562240430, 0.222405510587575];
q_t= 1; qd_t= 0; qdd_t = 0;
q_hat_next = q_t; qd_hat_next = qd_t;
figure();
hax1 = axes;
hold on;
surf(POS, VEL, VALUES, 'FaceAlpha',0.25)
contourf(POS, VEL, VALUES);
quiver(poses, vels, vq, vqd, 'b');
plot(q_hat_next, qd_hat_next, '-o');
title("J level sets");
xlabel("q");
ylabel("v");
pos_buff = [];
f_buff = [];
fnext = [q_hat_next; 0];
while sqrt([q_t, qd_t] * [q_t; qd_t]) > 0.01

    % compute the next state
%     qh_t2 = q_t + qd_t * 0.01; qdh_t2 = qd_t + qdd_t * 0.01;
%     plot(hax1, qh_t2, qdh_t2, '*');
    
    % assemble dxdt and project
    dfdt   = [qd_t; qdd_t];
    fnext  = proj_upper(q_t, qd_t, dfdt);
    f_buff = [f_buff; fnext'];

    pos_buff = [pos_buff; [q_hat_next, qd_hat_next, fnext(2)]];
    % integrate to find next state along V
    q_hat_next = q_t + qd_t  * 0.01;
    qd_hat_next = qd_t + fnext(2) * 0.01;
    
    % update state
    q_t = q_hat_next;
    qd_t = qd_hat_next;
    plot(hax1, q_hat_next, qd_hat_next, '+');
    drawnow;
end

figure();
ax1 = axes;
plot(ax1, pos_buff(:, 1), pos_buff(:, 2));

figure();
ax2 = axes;
hold on;
plot(ax2, pos_buff(:, 3));


function  x = relu(x)
    if x < 0
       x = 0;
    end
end