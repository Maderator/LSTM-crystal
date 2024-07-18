% NOT IMPLEMENTED BY US
% Copy of matlab method from https://www.mathworks.com/help/deeplearning/ug/initialize-learnable-parameters-for-custom-training-loop.html#mw_975f9595-ce05-4163-a5d7-1719d1609e0a
% to avoid problems with undefined initialize functions in the initialize
% method of the custom_layer classes.
% accessed on 2023-05-27

function parameter = initializeOrthogonal(sz)

Z = randn(sz,'single');
[Q,R] = qr(Z,0);

D = diag(R);
Q = Q * diag(D ./ abs(D));

parameter = dlarray(Q);

end