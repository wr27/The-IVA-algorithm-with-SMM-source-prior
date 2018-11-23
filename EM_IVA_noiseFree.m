function [x1 x2 nu1 pi1 nu2 pi2 W F] = EM_IVA_noiseFree(y1,y2,upS1,upS2,nu1,pi1,nu2,pi2,nEM)
% The EM code for IVA algorithm, No Noise! y=Wx

% Inputs:
%   y1(nK,nT)    : FFT coefficients for channel one of the mixed signal;
%   y2(nK,nT)    : FFT coefficients for channel two of the mixed signal;
%   upS1        : logic, upS1==1, update PDF for source 1;
%   upS2        : logic, upS2==1, update PDF for source 2;
%   nu1(K,nS)   : precision of prior for source 1;
%   pi1(nS,1)   : weights of GMM for prior of source 1;
%   nu2(K,nS)   : precision of prior for source 2;
%   pi2(nS,1)   : weights of GMM for prior of fsource 2;
%   nEM         : number of EM iterations;
% Outputs:
%   x1(nK,nT)    : estimated FFT coefficients of source 1;
%   x2(nK,nT)    : estimated FFT coefficients of source 2;
%   nu1(K,nS)   : precision of GMM for source 1;
%   pi1(nS,1)   : weights of GMM for source 1;
%   nu2(K,nS)   : precision of GMM for source 2;
%   pi2(nS,1)   : weights of GMM for source 2;
%   W(2,2,nK)   : W=inv(A), the DEmixing matrix, x=W*y;
%   F           : the likelihood value

% This code is the noise free version of the EM-IVA. 
[nK nT] = size(y1);
nS1 = size(nu1,2);
nS2 = size(nu2,2);
v = 4;              % Degree of freedom
% Random Initialization;
W = randn(2,2,nK);

% Unitary Initialization
% W(1,1,:) = 1;
% W(2,2,:) = 1;

  
% initialize by whitening matrix
% for k = 1:nK
%    W(:,:,k) = inv(sqrtm([y1(k,:);y2(k,:)]*[y1(k,:);y2(k,:)]'/nT));
% end

y11 = real(y1.*conj(y1));
y22 = real(y2.*conj(y2));
y12 = y1.*conj(y2);
Sig_11 = zeros(nS1*nS2,nK);
Sig_21 = zeros(nS1*nS2,nK);
Sig_22 = zeros(nS1*nS2,nK);
dS = zeros(nS1*nS2,1);

%%%%%%%%%%%%%%%%%%%%  EM STEP  %%%%%%%%%%%%%%%%%%
for iEM = 1:nEM
display(['Iteration:  ' num2str(iEM)]);   
W11nu1 = nu1.*repmat(abs(squeeze(W(1,1,:))).^2,1,nS1);
W21nu2 = nu2.*repmat(abs(squeeze(W(2,1,:))).^2,1,nS2);
W12nu1 = nu1.*repmat(abs(squeeze(W(1,2,:))).^2,1,nS1);
W22nu2 = nu2.*repmat(abs(squeeze(W(2,2,:))).^2,1,nS2);
W11W12nu1 = nu1.*repmat(squeeze(W(1,1,:)).*conj(squeeze(W(1,2,:))),1,nS1);
W21W22nu2 = nu2.*repmat(squeeze(W(2,1,:)).*conj(squeeze(W(2,2,:))),1,nS2);
for is1 = 1:nS1
    for is2 = 1:nS2
        is = (is2-1)*nS2+is1;
        Sig_11(is,:) = (W11nu1(:,is1)+W21nu2(:,is2))';
        Sig_21(is,:) = (W11W12nu1(:,is1)+W21W22nu2(:,is2)).';
        Sig_22(is,:) = (W12nu1(:,is1)+W22nu2(:,is2))';
        dS(is) = log(pi1(is1))+log(pi2(is2))+sum(log(nu1(:,is1).*nu2(:,is2)));
    end;
end;
f = repmat(dS,1,nT)-Sig_11*y11-2*real(Sig_21*y12)-Sig_22*y22;
maxf = max(f,[],1);
q = exp(f-repmat(maxf,nS1*nS2,1));
z = sum(q,1);
qs = q./repmat(z,nS1*nS2,1);

% Update A and source models.
sumqs = sum(qs,2);
sumq = reshape(sumqs,nS1,nS2);
qyy11 = ((v/2+nT/2)/v)*qs*y11';
qyy22 = ((v/2+nT/2)/v)*qs*y22';
qyy12 = ((v/2+nT/2)/v)*qs*y12.';
for k = 1:nK
    dnu = repmat(nu1(k,:)',nS2,1)-reshape(repmat(nu2(k,:),nS1,1),nS1*nS2,1);
    M11 = dnu'*qyy11(:,k);
    M12 = dnu'*qyy12(:,k);
    M22 = dnu'*qyy22(:,k);
    eig1 = (M11+M22)/2-sqrt((M11-M22)^2/4+abs(M12)^2);
    %eigvec = [M12/(eig1-M11);1]/sqrt(abs(M12/(eig1-M11))^2+1); 
    eigvec = [1; (eig1-M11)/M12]/sqrt(1+abs((eig1-M11)/M12)^2);
    %[V D] = eig([M11 M12; M12' M22]);
    %W2(:,:,k) = [conj(V(1,1)) conj(V(2,1)); -V(2,1) V(1,1)];
    W(:,:,k) = [conj(eigvec(1)) conj(eigvec(2)); -eigvec(2) eigvec(1)];
    %if (iEM>3); dbstop; end

    if upS1 && 0
        a1 = abs(W(1,1,k))^2*sum(reshape(qyy11(:,k),nS1,nS2),2);
        a2 = W(1,1,k)*conj(W(1,2,k))*sum(reshape(qyy12(:,k),nS1,nS2),2);
        a3 = abs(W(1,2,k))^2*sum(reshape(qyy22(:,k),nS1,nS2),2);
        nu1(k,:) = ((v/2+nT/2)/v)*((sum(sumq,2)./(a1+2*real(a2)+a3))');
        pi1 = sum(sumq,2); pi1 = pi1./sum(pi1);
end
   
   if upS2 && 0
        a1 = abs(W(2,1,k))^2*sum(reshape(qyy11(:,k),nS1,nS2),1);
        a2 = W(2,1,k)*conj(W(2,2,k))*sum(reshape(qyy12(:,k),nS1,nS2),1);
        a3 = abs(W(2,2,k))^2*sum(reshape(qyy22(:,k),nS1,nS2),1);
        nu2(k,:) = ((v/2+nT/2)/v)*((sum(sumq,1)./(a1+2*real(a2)+a3)));
        pi2 = sum(sumq,1)'; pi2 = pi2./sum(pi2);
    end
end;

F(iEM) = sum(maxf)+sum(log(z));
plot(F); %drawnow; 
end

%%%% Signal Estimation
x1 = repmat(squeeze(W(1,1,:)),1,nT).*y1+repmat(squeeze(W(1,2,:)),1,nT).*y2;
x2 = repmat(squeeze(W(2,1,:)),1,nT).*y1+repmat(squeeze(W(2,2,:)),1,nT).*y2;
return;