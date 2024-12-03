% Definindo as matrizes A, B, C, D
A = [ 1.1, -1.11, 0, 0;
      1.11, -2.2, 0.11, 0;
      0, 0.11, 0.44, 0;
      0.06 / (0.694*8*2.23), 0, -0.06 / (0.694*8*2.23), 0 ];

B = [-0.008, -6;
     -0.008, 0;
     -0.008, -6;
     -0.008, 0];    

C = [1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, 0];

D = [-0.008, 0.001;
     -0.008, 0.001;
     -0.008, 0.001];

% Definindo a variável simbólica s
s = tf('s');

% Criar a matriz identidade multiplicada por s
I = eye(size(A));

% Calcular (sI - A)
sI_minus_A = s*I - A;

% Calcular a inversa de (sI - A)
inv_sI_minus_A = inv(sI_minus_A);
G
% Calcular a função de transferência: G(s) = C(sI - A)^-1 * B + D
G = C * inv_sI_minus_A * B + D;
% G é a matriz de funções de transferência
G11 = G(1, 1);  % Função de transferência da 1ª saída para a 1ª entrada
bode(G);
grid on;
]