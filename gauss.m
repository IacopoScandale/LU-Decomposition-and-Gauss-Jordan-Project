%% Algoritmo di Gauss per Fattorizzazione LU, Applicazioni e Running Time
clear
clc
close all


%% è corretto LU?
A = rand(10,10);
b = rand(10,1);

x = A\b;

% PA = LU
% PAx = Pb
% PLUx = Pb
% Risolvo 2 sistemi:
% Ly = z and Ux = y
[L,U,P] = my_lu_pivoting(A);
b1 = P*b;
y1 = L\b1;
x1 = U\y1;

x-x1




%% Running Time Fattorizzazione LU 
r = 50;
range = 1:r;
% matrici random

    % t_my_lu = zeros(1,r);
    % t_lu_mat = zeros(1,r);
    
    % for i = range
    %     disp(i)
    %     n = i*50;
    %     A = rand(n,n);
    % 
    %     tic
    %     [~,~,~] = my_lu_pivoting(A);
    %     t_my_lu(i) = toc;
    % 
    %     tic
    %     [~,~,~] = lu(A);
    %     t_lu_mat(i) = toc;
    %     clc
    % end
    
    % save('t_my_lu.mat','t_my_lu')
    % save('t_lu_mat.mat','t_lu_mat')

t_my_lu = load('t_my_lu.mat').t_my_lu;
t_lu_mat = load('t_lu_mat.mat').t_lu_mat;


% Matrici Sparse (Poisson)

    % t_my_lu_sparse = zeros(1,r);
    % t_lu_mat_sparse = zeros(1,r);
    
    % for i = range
    %     disp(i)
    %     n = i*50;
    %     A = gallery('poisson',floor(sqrt(n)));
    %     disp(size(A))
    % 
    %     tic
    %     [~,~,~] = my_lu_pivoting(A);
    %     t_my_lu_sparse(i) = toc;
    % 
    %     tic
    %     [~,~,~] = lu(A);
    %     t_lu_mat_sparse(i) = toc;
    %     clc
    % end
    
    % save('t_my_lu_sparse.mat','t_my_lu_sparse')
    % save('t_lu_mat_sparse.mat','t_lu_mat_sparse')

t_my_lu_sparse = load('t_my_lu_sparse.mat').t_my_lu_sparse;
t_lu_mat_sparse = load('t_lu_mat_sparse.mat').t_lu_mat_sparse;


% Matrici di hilbert
    % t_my_lu_hil = zeros(1,r);
    % t_lu_mat_hil = zeros(1,r);
    % 
    % for i = range
    %     disp(i)
    %     n = i*50;
    %     A = hilb(n);
    %     disp(size(A))
    % 
    %     tic
    %     [~,~,~] = my_lu_pivoting(A);
    %     t_my_lu_hil(i) = toc;
    % 
    %     tic
    %     [~,~,~] = lu(A);
    %     t_lu_mat_hil(i) = toc;
    %     clc
    % end
    % 
    % save('t_my_lu_hil.mat','t_my_lu_hil')
    % save('t_lu_mat_hil.mat','t_lu_mat_hil')

t_my_lu_hil = load('t_my_lu_hil.mat').t_my_lu_hil;
t_lu_mat_hil = load('t_lu_mat_hil.mat').t_lu_mat_hil;




% plotting
figure;

subplot(1, 2, 1);
loglog(range, t_my_lu, 'b-', 'LineWidth', 2);
hold on;
loglog(range, range .^ 3, 'r--', 'LineWidth', 2);
loglog(range, t_lu_mat)
loglog(t_my_lu_sparse)
loglog(t_lu_mat_sparse)
loglog(t_my_lu_hil)
loglog(t_lu_mat_hil)
legend('my random', 'O(n³)','matlab random','my poisson','matlab poisson','my hilbert','matlab hilbert', 'Location', 'northwest');
title('LogLog Running Time')

subplot(1, 2, 2);
semilogy(range, t_my_lu, 'b-', 'LineWidth', 2);
hold on;
semilogy(range, range .^ 3, 'r--', 'LineWidth', 2);
semilogy(range, t_lu_mat)
semilogy(t_my_lu_sparse)
semilogy(t_lu_mat_sparse)
semilogy(t_my_lu_hil)
semilogy(t_lu_mat_hil)
legend('my random', 'O(n³)','matlab random','my poisson','matlab poisson','my hilbert','matlab hilbert', 'Location', 'northwest');
title('Semilog Running Time')

sgtitle('Running Time Fattorizzazione LU');

%% Fattorizzaione LU su Matrici con Determinante Nullo
A = rand(10);

% Duplico una Riga
A(2, :) = A(1, :);

det(A)

% [L,U] = my_lu(A)
% [L,U,~,~,rg] = my_lu_pivoting(A)
% [L,U] = lu(A)

%% abbasso ulteriormente il rango della matrice
A(3, :) = A(1, :);
A(4, :) = A(1, :);

% [L,U,~,~,rg] = my_lu_pivoting(A)
% [L,U] = lu(A)





%% determinante con LU

    % t_det_lu = zeros(r);
    % t_det_matlab = zeros(r);

    % for i = range
    %     disp(i)
    %     n = i*50;
    %     A = rand(n,n);
    % 
    %     tic
    %     my_det(A);
    %     t_det_lu(i) = toc;
    % 
    %     tic
    %     det(A);
    %     t_det_matlab(i) = toc;
    %     clc
    % end
    
    % save('t_det_lu.mat','t_det_lu')
    % save('t_det_matlab.mat','t_det_matlab')

t_det_lu = load('t_det_lu.mat').t_det_lu;
t_det_matlab = load('t_det_matlab.mat').t_det_matlab;

% plotting
figure;
subplot(1,2,1)
loglog(t_det_lu)
hold on;
loglog(t_det_matlab)
legend('t det random','t det matlab random', 'Location', 'northwest');

subplot(1,2,2)
semilogy(t_det_lu)
hold on;
semilogy(t_det_matlab)
legend('t det random','t det matlab random', 'Location', 'northwest');


%% sistemi lineari

    % t_my_lu_Ab = zeros(1,r);
    % t_mat_lu_Ab = zeros(1,r);
    % t_mat_Ab = zeros(1,r);
    % 
    % for i=range
    %     disp(i)
    %     n = i*50;
    %     A = rand(n);
    %     b = rand(n,1);
    % 
    %     tic
    %     x = A\b;
    %     t_mat_Ab(i)=toc;
    % 
    %     tic
    %     [L,U,P] = my_lu_pivoting(A);
    %     b1 = P*b;
    %     y = L\b1;
    %     x = U\y;
    %     t_my_lu_Ab(i)=toc;
    % 
    %     tic
    %     [L,U,P] = lu(A);
    %     b2 = P*b;
    %     y = L\b2;
    %     x = U\y;
    %     t_mat_lu_Ab(i)=toc;
    %     clc
    % end
    % 
    % save('t_my_lu_Ab.mat','t_my_lu_Ab')
    % save('t_mat_Ab.mat','t_mat_Ab')
    % save('t_mat_lu_Ab.mat','t_mat_lu_Ab')

t_my_lu_Ab = load('t_my_lu_Ab.mat').t_my_lu_Ab;
t_mat_Ab = load('t_mat_Ab.mat').t_mat_Ab;
t_mat_lu_Ab = load('t_mat_lu_Ab.mat').t_mat_lu_Ab;


% plotting
figure;

subplot(1, 2, 1);
loglog(t_my_lu_Ab, 'b-', 'LineWidth', 2);
hold on;
loglog(range, range .^ 3, 'r--', 'LineWidth', 2);
loglog(t_mat_Ab)
loglog(t_mat_lu_Ab)

legend('my lu A\b', 'O(n³)','matlab A\b','matlab lu A\b', 'Location', 'northwest');
title('LogLog Running Time Linear System')

subplot(1, 2, 2);
semilogy(t_my_lu_Ab, 'b-', 'LineWidth', 2);
hold on;
semilogy(range, range .^ 3, 'r--', 'LineWidth', 2);
semilogy(t_mat_Ab)
semilogy(t_mat_lu_Ab)

legend('my lu A\b', 'O(n³)','matlab A\b','matlab lu A\b', 'Location', 'northwest');
title('Semilog Running Time Linear System')

sgtitle('Running Time Linear System with LU');




%% Gauss-Jordan
m = 50;
n = 50;

A = rand(m,n);
b = rand(m,1);
x = A\b;


R1 = rref([A,b]);
A1 = R1(:,1:n);
b1 = R1(:,n+1);
x1 = A1\b1;


R2 = my_gauss_jordan([A,b]);
A2 = R2(:,1:n);
b2 = R2(:,n+1);
% x2 = A2\b2;

% x1-x2
x1-b2


figure;
subplot(1,2,1)
spy(R1)
title('matlab G-J')
subplot(1,2,2)
spy(R2)
title('my G-J')

%% Esempi
A = rand(5,10)
R = rref(A)
spy(R)

%% 
A = rand(7,3)
R = rref(A)
spy(R)

%% sistemi quadrati
n = 2100;
m = 2100;
A = rand(m,n);
b = rand(m,1);

tic
x = A\b;
toc


tic
R1 = rref([A,b]);
toc

% >> Elapsed time is 0.100767 seconds.
% >> Elapsed time is 41.437204 seconds.

x1 = R1(:,n+1);
x-x1;


%% Applicazione Gauss-Jordan per il calcolo dell'inversa
n = 10;
A = rand(n);
AI = [A,eye(n)];

IAinv = rref(AI);

Ainv = IAinv(:,n+1:end);

norm(Ainv*A - eye(n))



% cfr: è esattamente come risolvere n sistemi lineari con b = e₁ ... eₙ

%% Fill in
% Dimensione della matrice
n = 100;

% Valori delle diagonali
main_diag = 4 * ones(n, 1);
diag = 1* ones(n, 1);

% Metti insieme le diagonali con le lunghezze corrette
diagonals = [diag,diag main_diag,diag diag];
diags = [diag,main_diag,diag];

% Indici delle diagonali (-1, 0, 1 per inferiore, principale, superiore)
diag_indices = [-40,-1, 0, 1, 40];
diags_indices = [-40,0,40];

% Crea la matrice a bande usando spdiags
A = spdiags(diagonals, diag_indices, n, n);
B = spdiags(diags,diags_indices,n,n);

[L,U,P] = lu(A);
[LL,UU,PP]=lu(B);

A_poiss = full(gallery('poisson',10));
[L_poiss,U_poiss,~] = lu(A_poiss);


% plotting
figure;
subplot(3, 3, 1);
spy(A)
title('A')
subplot(3, 3, 2);
spy(L)
title('L')
subplot(3, 3, 3);
spy(U)
title('U')

subplot(3, 3, 4);
spy(B)
title('B')
subplot(3, 3, 5);
spy(LL)
title('L')
subplot(3, 3, 6);
spy(UU)
title('U')

subplot(3, 3, 7);
spy(A_poiss)
title('A poisson')
subplot(3, 3, 8);
spy(L_poiss)
title('L')
subplot(3, 3, 9);
spy(U_poiss)
title('U')


%% Gauss Fattorizzazione LU (senza pivoting)
function [L,U] = my_lu(A)
    % Verifica che A sia una matrice quadrata
    [n, m] = size(A);
    if n ~= m
        error('La matrice A deve essere quadrata');
    end

    U = A;
    L = eye(n);
    
    for k = 1:n-1
        for i = k+1:n
            % caso pivot nullo
            if U(k,k)==0
                error('error: chosen pivot is zero')
            end
            L(i,k) = U(i,k) / U(k,k);
            U(i, k:n) = U(i, k:n) - L(i, k) * U(k, k:n);        
        end
    end

end





%% Fattorizzazione LUP, Gauss con pivoting del max

function [L, U, P, scambi, rango] = my_lu_pivoting(A)
    % Verifica che A sia una matrice quadrata
    [n, m] = size(A);
    if n ~= m
        error('La matrice A deve essere quadrata');
    end
    
    % Inizializzazione delle matrici L, U e P
    L = eye(n);
    U = A;
    P = eye(n);
    scambi = 0;
    rango = n;
    
    % Fattorizzazione LU con pivoting
    for k = 1:n-1
        % Pivoting: indice del massimo elemento nella colonna k a partire da k
        [M, maxIndex] = max(abs(U(k:n, k)));
        
        % diventa ora indice di riga della matrice U
        maxIndex = maxIndex + k - 1;

        % caso det A == 0
        if M == 0
            % Aggiungo epsilon per evitare NaN
            U(maxIndex,k) = U(maxIndex,k) + 1e-6;
            rango = rango - 1;
        end
        
        % Scambia la riga corrente con la riga del massimo elemento in U
        if maxIndex ~= k
            scambi = scambi + 1;
            U([k, maxIndex], :) = U([maxIndex, k], :);
            % Permuto anche l'identità
            P([k, maxIndex], :) = P([maxIndex, k], :);
            % scambiare anche in L per mantenere PA=LU
            if k > 1
                L([k, maxIndex], 1:k-1) = L([maxIndex, k], 1:k-1);
            end
        end
        
        % Elimina gli elementi sotto il pivot e aggiorna L e U
        for i = k+1:n
            L(i, k) = U(i, k) / U(k, k);
            U(i, k:n) = U(i, k:n) - L(i, k) * U(k, k:n);
        end
    end
    if U(end,end) == 0
        rango = rango-1;
    end
end





%% Applicazione calcolo del determinante con la fattorizzazione LU
% calcolo del determinante con lo stesso metodo di matlab, ovvero
% sfruttando la fattorizzazione lu

function [det] = my_det(A)

    [~,U,~,scambi] = my_lu_pivoting(A);
    det = prod(diag(U)) * (-1)^scambi;
end






%% Algoritmo di Gauss-Jordan
function R = my_gauss_jordan(A)
    % Ottiene le dimensioni della matrice A
    [m, n] = size(A);

    % Ciclo su ogni colonna
    for k = 1:min(m,n)
        % Pivoting per selezionare l'elemento massimo della colonna sotto
        % la riga k inclusa
        [~, i_max] = max(abs(A(k:m, k)));
        i_max = i_max + k - 1;

        % Scambia le righe se necessario
        if i_max ~= k
            A([k, i_max], :) = A([i_max, k], :);
        end

        % Se l'elemento pivot è zero, continua alla colonna successiva
        % (prossima iterazione del for)
        if A(k, k) == 0
            continue;
        end

        % Normalizza la riga del pivot, trasformando il pivot in un 1
        A(k, :) = A(k, :) / A(k, k);

        % Elimina gli elementi sopra e sotto il pivot
        for i = 1:m
            if i ~= k
                A(i, :) = A(i, :) - A(i, k) * A(k, :);
            end
        end
    end

    R = A; % La matrice R è la matrice ridotta a scala risultante
end