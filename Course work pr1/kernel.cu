#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <windows.h>
#include <intrin.h>

using namespace std;

#define blockX 32
#define blockY 16

__global__ void kernelGPU(int* a, int* b, int rows, int cols) {

	int x = blockIdx.x * blockDim.x;
	//blockDim – размер блока; blockIdx – индекс текущего блока в сетке;
	int y = blockIdx.y * blockDim.y;
	int i = threadIdx.x + x;
	//threadIdx – индекс текущей нити в блоке;
	int j = threadIdx.y + y;
	if ((i < rows) && (j < cols)) {
		b[i * cols + j] = a[j * rows + i];
	}
}

//Сравнение матриц
int compareMatrix(int* a, int* b, int rows, int cols) {
	for (int i = 0; i < rows * cols; i++) {
		if (a[i] != b[i]) {
			cout << "	!" << endl;
			return 1;
		}
	}
	cout << "Матрицы GPU и CPU равны!" << endl;
	return 0;
}

//Трансконирование матрицы
int CPUfunction(int* a, int* b, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			b[i * cols + j] = a[j * rows + i];
		}
	}
	return 0;
}

//Вывод информации об устройстве
static void deviceInfo() {
	cout << "Информация об используемых устройствах:" << endl;
	//CPU
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];

	char CPUBrandString[0x40] = { 0 };
	for (unsigned int i = 0x80000000; i <= nExIds; ++i)
	{
		__cpuid(CPUInfo, i);

		if (i == 0x80000002)
		{
			memcpy(CPUBrandString,
				CPUInfo,
				sizeof(CPUInfo));
		}
		else if (i == 0x80000003)
		{
			memcpy(CPUBrandString + 16,
				CPUInfo,
				sizeof(CPUInfo));
		}
		else if (i == 0x80000004)
		{
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
		}
	}
	cout << "Процессор: " << CPUBrandString << endl;
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo);
	cout << "Количество ядер: " << siSysInfo.dwNumberOfProcessors << endl;

	//GPU
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; ++i)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		cout << "Видеокарта: " << deviceProp.name << endl;
		cout << "Полная глобальная память: " << deviceProp.totalGlobalMem << " байт" << endl;
		cout << "Максимальное количество потоков на блок: " << deviceProp.maxThreadsPerBlock << endl;
		cout << "Максимальное количество потоков на мультипроцессор: " << deviceProp.maxThreadsPerMultiProcessor << endl;
		cout << "Максимальные размеры сетки: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << endl;
		cout << "Максимальные размеры блока: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl;
		//cout << "Общая память на блок: " << deviceProp.sharedMemPerBlock << endl;
	}

}

// Вывод созданных и используемых в программе матриц для проверки  
void showMatrix(int* a, int choose, int rows, int cols) {
	if (choose == 1) {
		//Вывод стартовой матрицы на экран(10x10)
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				cout << setw(2) << a[j * rows + i] << " ";
			}
			cout << endl;
		}
	}
	else {
		//Вывод транспонированных матриц на экран(10x10)
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				cout << setw(2) << a[j * cols + i] << " ";
			}
			cout << endl;
		}
	}
	cout << endl;
}

static void Info(int rows, int cols) {
	cout << "--------------------------------" << endl;
	cout << "Текущий размер матрицы:" << rows << "x" << cols << endl;
}


int main() {

	setlocale(LC_ALL, "Rus");
	
	int cols = 10;
	int rows = 10;

	int* hostA, * hostB, * hostC;
	int* devA, * devB;
	
	//Создание массива(таблицы) зависимости времени от размера матрицы
	string output[8][4];
	output[0][0] = { "Размер матрицы:    " };
	output[0][1] = { "Время CPU:         " };
	output[0][2] = { "Время GPU:         " };
	output[0][3] = { "Отношение СPU/GPU: " };

	for (int k = 1; k <= 7;) {

		//Выделение памяти 
		hostA = (int*)malloc(sizeof(int) * cols * rows);
		hostB = (int*)malloc(sizeof(int) * cols * rows);
		hostC = (int*)malloc(sizeof(int) * cols * rows);

		for (int i = 0; i < cols; i++) {	
			for (int j = 0; j < rows; j++) {
					hostA[i * rows + j] = (int)rand() % 100;
			}
		}

		// Выделение памяти на GPU.
		cudaMalloc(&devA, sizeof(int) * rows * cols);
		cudaMalloc(&devB, sizeof(int) * rows * cols);

		// Скопировать входные данные из памяти CPU в память GPU.
		cudaMemcpy(devA, hostA, sizeof(int) * rows * cols, cudaMemcpyHostToDevice);

		// Задать конфигурацию запуска n нитей.
		dim3 blockDim = dim3(blockX, blockY);
		dim3 gridDim = dim3((rows + 31) / 32, (cols + 15) / 16);

		// Объявление переменных-событий начала и окончания
		// Используем cudaEvent для замеров времени
		cudaEvent_t gpuStart, gpuStop;

		// Инициализация переменных-событий.
		cudaEventCreate(&gpuStart);
		cudaEventCreate(&gpuStop);
		// Привязка события start к данной позиции в коде
		// программы (начало выполнения ядра).
		cudaEventRecord(gpuStart, 0);

		// Запуск GPU-ядра.
		kernelGPU << < gridDim, blockDim >> > (devA, devB, rows, cols);
		// Привязка события stop к данной позиции в коде
		// программы (окончание выполнения ядра).
		cudaEventRecord(gpuStop, 0);
		// Ожидание окончания выполнения ядра,
		// синхронизация по событию stop.
		cudaEventSynchronize(gpuStop);
		float elapsedTime;
		// Получение времени, прошедшего между событиями start и stop.
		cudaEventElapsedTime(&elapsedTime, gpuStart, gpuStop);
		
		//Вызов вывода текущего состояния
		Info(rows, cols);

		cout << "GPU time: " << elapsedTime << "ms" << endl;
		
		// Скопировать результаты в память CPU.
		cudaMemcpy(hostB, devB, sizeof(int) * cols * rows, cudaMemcpyDeviceToHost);

		//Повторение вычислений на процессоре
		auto start = std::chrono::high_resolution_clock::now();

		CPUfunction(hostA, hostC, rows, cols);

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = stop - start;
		
		cout << "CPU time: " << time.count() * 1000 << "ms" << endl;

		//Формирование таблицы вывода результатов
		string ratio = to_string(time.count() * 1000 / elapsedTime);
		output[k][0] = {"| " + to_string(rows) + "x" + to_string(cols) + " "};
		output[k][1] = {"| " + to_string(time.count() * 1000) + "ms " };
		output[k][2] = {"| " + to_string(elapsedTime) + "ms " };
		output[k][3] = {"| " + ratio + " " };

		
		cout << endl << "StartMatrix:" << endl;
		showMatrix(hostA, 1, rows, cols);
		cout << "GPU:" << endl;
		showMatrix(hostB, 2, rows, cols);
		cout << "CPU:" << endl;
		showMatrix(hostC, 2, rows, cols);
		

		//Сравнение матриц
		compareMatrix(hostB, hostC, rows, cols);

		//Очистка созданной динамической памяти
		free(hostA);
		free(hostB);
		free(hostC);
		// Освободить выделенную память GPU.
		cudaFree(devA);
		cudaFree(devB);

		k++;

		//Задание размера матриц 10, 50, 100, 500, 1000, 5000, 10000
		if (k % 2 == 0) {
			cols *= 5;
			rows *= 5;
		}
		else {
			cols = cols * 2;
			rows = rows * 2;
		}
	}
	cout << endl;

	deviceInfo();

	//Вывод таблицы на экран
	cout << endl;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 8; j++) {
			cout << output[j][i] << "\t";
		}
		cout << endl;
	}
	return 0;
}