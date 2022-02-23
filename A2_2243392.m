%% Restoration for noisy1 image
%Clear all before start
clear all;clc;
%read my noisy1 image
myImage=imread('noisy1.png');
%show my image
figure;imshow(myImage);title('Original Image');
%find row and column of my Image, size
[M, N]=size(myImage);

%When we look at the image, we see that there is a periodic type noise in Frequency Domain,
%so examining the histogram does not help us in this regard.
%Therefore, we need to look at the Fourier transform and move on.
%Conclusion, I will use the Ideal Notch Reject filter to correct periodic type noise.

% Fourier transform
FT = fft2(myImage)/(M*N);
% Centred fourier spectrum
CFS = fftshift(FT);
F=fftshift(fft2(myImage));
%show my Fourier transform
figure;imshow(CFS);title('Fourier Spectrum');
%When we examine the fourier spectrum,
%we can easily see the noises and their locations, and we can use the Ideal notch filter to remove them.
%for Notch Filter
%Define the transfer function of order n 
n = 2;
% Define cutoff frequency D0
D0=120;
uk=-70;
vk=80;
for u=1:M
    for v=1:N
        Dkp(u,v) = ((u-(M/2)-uk)^2 + (v-(N/2)-vk)^2 ) ^ ( 1/2);
        Dkn(u,v) = ((u-(M/2)+uk)^2 + (v-(N/2)+vk)^2 ) ^ ( 1/2);
    end
end
%for Ideal Notch Filter
H = zeros(M,N);
H(Dkp<=D0) = 1; Hp = 1-H;
H = zeros(M,N);
H(Dkn<=D0) = 1; Hn = 1-H;
Hnr = Hp.*Hn; Hnp = 1-Hnr;

%show my filter
figure;imshow(Hnp,[]);
%Filter multiplied by the Fourier transform
I=Hnp.*F;
%Inverse Fourier transform 
Im=abs((ifft2((I))));
%for my output image
Reconstructed=uint8(255*mat2gray(Im));

%write my Recovered Image
imwrite(Reconstructed,'recovered1.png');
%show my Recovered Image
figure, imshow(Im,[]); title('Reconstructed Image');

%Differences between edges of noisy and reconstructed images
%find edge of the original image
edge_original = edge(myImage);
%find edge of the reconstructed image
edge_reconstructed = edge(Reconstructed);
%show edge of the original image
figure, imshow(edge_original);title('Edge of Original Image');
%show edge of the reconstructed image
figure, imshow(edge_reconstructed);title('Edge of Reconstructed Image');

%% Restoration for noisy2 image
%Clear all before start
clear all;clc;
%read my noisy2 image
myImage_2=imread('noisy2.png');
%show my image
figure;imshow(myImage_2);title('Original Image');

%Take a block in the image to examine the histogram
block=myImage_2(1060:1140,5:75);
figure;imshow(block);title('Block');

%Histogram of image
%find row and column of my block
rows = size(block,1); columns = size(block,2);
%create an empty array with 256 elements for finding how many have the same value
hist = zeros(256,1); 

for i=1:rows %loop for rows of matrix
    for j=1:columns %loop for columns of matrix
        %how many have the same value
        %find the frequency of each pixels
        for k=0:255
            if(block(i,j) == k)
               value = block(i,j); %this is my value which is i,j index
               hist(value+1)=hist(value+1)+1;  %write the value to the empty hist array 
            end
        end
    end
end
%show my histogram of part of the noisy image
x_axis = (0:1:255);
figure;bar(x_axis,hist);title('Histogram of the block');

%Spatial Domain
%When we look at the histogram, we can see that it is a Uniform noise from Additive type noise - Spatially Independent.

%The mean of the density is
meanb = mean(block(:));
%The variance of the density is
varb = var(double(block(:)));
x_axis=[0:255];hold on

pz = zeros(256,1);
b = (sqrt(12*varb) + 2*meanb)/2;
a = (2*meanb) - b;
%Form the Uniform noise
for i=1:256
    if a <= x_axis(i) && x_axis(i) <= b
        pz(i) = 1/(b-a);
    else 
        pz(i) = 0;
    end  
end
ng=pz*sum(hist)/sum(pz);
%show the plot
plot(x_axis,ng,'r-','linewidth',2)

% When we crop a part of the noisy image and examine the histogram of that part and plot,
%it shows us that there is a Uniform type noise in our image,
%so I will use the midpoint filter from the Order Statistics Filters as the best filter for Uniform type noises.

%Remove the noise from the image using Midpoint Filter

%initializa my array
mask=zeros(3,3);
%for my output image
Output=uint8(zeros(size(myImage_2,1),size(myImage_2,2))); 

%find row and column of my original image
row = size(myImage_2,1); column = size(myImage_2,2);

for i=2:row-1
    for j=2:column-1
        mask(1,1) = myImage_2(i-1,j-1);
        mask(1,2) = myImage_2(i-1,j);
        mask(1,3) = myImage_2(i-1,j+1);
        mask(2,1) = myImage_2(i,j-1);
        mask(2,2) = myImage_2(i,j);
        mask(2,3) = myImage_2(i,j+1);
        mask(3,1) = myImage_2(i+1,j-1);
        mask(3,2) = myImage_2(i+1,j);
        mask(3,3) = myImage_2(i+1,j+1);
        
        arr = [mask(1,:),mask(2,:),mask(3,:)];
        
        %for midpoint filter (max+min)/2
        maxi = max(arr);
        mini = min(arr);
        avg = (maxi + mini)/2;
        Output(i,j)= avg;
        
    end
end


%write my Recovered Image
imwrite(Output,'recovered2.png');
%show my Recovered Image
figure, imshow(Output); title('Reconstructed Image');

%Differences between edges of noisy and reconstructed images
%find edge of the original image
edge_original = edge(myImage_2);
%find edge of the reconstructed image
edge_reconstructed = edge(Output);
%show edge of the original image
figure, imshow(edge_original);title('Edge of Original Image');
%show edge of the reconstructed image
figure, imshow(edge_reconstructed);title('Edge of Reconstructed Image');

%% Restoration for noisy3 image
clear all;clc;
%read my noisy3 image
myImage_3=imread('noisy3.png');
%show my original image
figure;imshow(myImage_3);title('Original Image');

%Take a block in the image to examine the histogram
block2=myImage_3(310:357,133:151);
%show the cropped block from the image 
figure;imshow(block2);title('Block');

%Histogram of image
%find row and column of my block
rows_2 = size(block2,1); columns_2 = size(block2,2);
%create an empty array with 256 elements for finding how many have the same value
hist2 = zeros(256,1); 
value = 0;
for i=1:rows_2 %loop for rows of matrix
    for j=1:columns_2 %loop for columns of matrix
        %how many have the same value
        %find the frequency of each pixels
        for k=0:255
            if(block2(i,j) == k)
               value = block2(i,j); %this is my value which is i,j index
               hist2(value+1)=hist2(value+1)+1;  %write the value to the empty hist array 
            end
        end
    end
end
%show my histogram of part of the noisy image
x_axis2 = (0:1:255);
figure;bar(x_axis2,hist2);title('Histogram of the Block');
%Spatial Domain
%We can see that it is a Salt noise from Additive type noise - Spatially Independent.

%When we look at this image in other words if we make visual inspection,
%we can clearly see that there are salt noises in the image,
%and our one of the best option for this type of noise is the Median filter,
%which is also one of the Order Statistics Filters.

%Remove the noise from the image using Median Filter

%initializa my array
arr=[0 0 0 ; 0 0 0 ; 0 0 0];
%for my output image
Output2=uint8(zeros(size(myImage_3,1),size(myImage_3,2))); 

%find row and column of my block
row = size(myImage_3,1); column = size(myImage_3,2);

for i=2:row-1
    for j=2:column-1
        arr=[myImage_3(i-1,j-1),myImage_3(i-1,j),myImage_3(i-1,j+1),myImage_3(i,j-1),myImage_3(i,j),myImage_3(i,j+1),myImage_3(i+1,j-1),myImage_3(i+1,j),myImage_3(i+1,j+1)];
        %for median filter median(arr)
        Output2(i,j)= median(arr);
        arr=[];
    end
end

%write my Recovered Image
imwrite(Output2,'recovered3.png');
%show my Recovered Image
figure, imshow(Output2); title('Reconstructed Image');

%Differences between edges of noisy and reconstructed images
%find edge of the original image
edge_original = edge(myImage_3);
%find edge of the reconstructed image
edge_reconstructed = edge(Output2);
%show edge of the original image
figure, imshow(edge_original);title('Edge of Original Image');
%show edge of the reconstructed image
figure, imshow(edge_reconstructed);title('Edge of Reconstructed Image');


