#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>
#include <deque>
#include <time.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <omp.h>

using namespace std;
using namespace Eigen;

class Factors
	/* initial a tick string to the factor (without normalization) */
{
public:
	struct tickData
	{
		//string market;
		//string ticker;
		string time;
		double lastPrice;
		int openInterest;
		int openInterestDelta;
		//int tradeMoney;
		int tradeVol;
		//int opening;
		//int closing;
		double bidPrice;
		double askPrice;
		int bidVol;
		int askVol;
	};
	tickData tick;
	//double factor[7];
	double factor[6];
	//double lastPrice;
	//int openInterestDelta;
	//int tradeVol;           
	//double stdSpread;
	//double stdMean;
	//double depth;
	//double volRatio;

	Factors(string& csvstr)
	{
		tick = strToTick(csvstr);
		//factor[0] = tick.lastPrice;
		factor[0] = log((tick.askPrice - tick.bidPrice) / tick.lastPrice);
		factor[1] = log((tick.askPrice + tick.bidPrice) / 2.0 / tick.lastPrice);
		factor[2] = tick.tradeVol;
		factor[3] = log(double(tick.bidVol) / double(tick.askVol));
		factor[4] = tick.askVol + tick.bidVol;
		factor[5] = tick.openInterestDelta;
	}
private:
	tickData strToTick(string& csvstr) {
		vector<string> mData;
		mData = csvToVector(csvstr);
		tickData data;
		data.time = mData[2];
		data.lastPrice = stod(mData[3]);
		data.openInterest = stoi(mData[4]);
		data.openInterestDelta = stoi(mData[5]);
		data.tradeVol = stoi(mData[7]);
		data.bidPrice = stod(mData[12]);
		data.askPrice = stod(mData[13]);
		data.bidVol = stoi(mData[14]);
		data.askVol = stoi(mData[15]);
		return data;
	}

	vector<string> csvToVector(const string& csvstr) {
		stringstream lineStream(csvstr);
		string cell;
		vector<string> mData;
		while (getline(lineStream, cell, ','))
		{
			mData.push_back(cell);
		}
		return mData;
	}
};

vector<vector<double>> readCSV(const string& filename)
{
	ifstream infile(filename);
	string buffer;
	vector<vector<double>> data;
	while (getline(infile, buffer, '\n'))
	{
		stringstream lineStream(buffer);
		string cell;
		vector<double> mData;
		while (getline(lineStream, cell, ','))
		{
			mData.push_back(stod(cell));
		}
		data.push_back(mData);
	}
	return data;
}

MatrixXf toMatrix(vector<vector<double>>& vec)
{
	int rows = vec.size();
	int cols = vec[0].size();
	MatrixXf matrix(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrix(i, j) = vec[i][j];
		}
	}
	return matrix;
}

class NeuralNet
{
public:
	struct Weights
	{
		MatrixXf weight0;
		MatrixXf weight1;
		MatrixXf weight2;
		MatrixXf bias0;
		MatrixXf bias1;
		MatrixXf bias2;
	};
	Weights weights;
	//RowVectorXf input;
	ArrayXXf prediction;

	NeuralNet(const string weightFiles[], const string biasFiles[])
	{
		vector<vector<double>> weight;
		vector<vector<double>> bias;

		// input layer -> first hidden layer
		weight = readCSV(weightFiles[0]);
		weights.weight0 = toMatrix(weight);
		bias = readCSV(biasFiles[0]);
		weights.bias0 = toMatrix(bias);

		// first hidden layer -> second hidden layer
		weight = readCSV(weightFiles[1]);
		weights.weight1 = toMatrix(weight);
		bias = readCSV(biasFiles[1]);
		weights.bias1 = toMatrix(bias);

		// second hidden layer -> output layer
		weight = readCSV(weightFiles[2]);
		weights.weight2 = toMatrix(weight);
		bias = readCSV(biasFiles[2]);
		weights.bias2 = toMatrix(bias);
	}

	ArrayXXf predict(RowVectorXf input)
	{
		MatrixXf a1 = input * weights.weight0 + weights.bias0.transpose();
		ArrayXXf c1 = tanh(a1.array());
		MatrixXf a2 = c1.matrix() * weights.weight1 + weights.bias1.transpose(); 
		ArrayXXf c2 = tanh(a2.array());
		MatrixXf a3 = c2.matrix() * weights.weight2 + weights.bias2.transpose();
		prediction = softmax(a3.array());

		return prediction;
	}
private:
	ArrayXXf tanh(ArrayXXf x)
	{
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}
	ArrayXXf softmax(ArrayXXf x)
	{
		ArrayXXf e = exp(x);  // do not subtract max(x) right now
		double s = e.sum();
		return e / s;
	}
};


class Strategy
{
public:
	struct Transaction
	{
		string direction;
		string openTime;
		double openPrice;
		//double openNextPrice;
		string closeTime;
		double closePrice;
		//double closeNextPrice;
		double profit;
		double yield;
	};
	Transaction longRecord;
	Transaction shortRecord;
	Transaction emptyRecord;
	//double score[2];
	double trigger[2];
	double interest;
	//int multiplicator;
	double loss;
	Factors::tickData tick;
	bool long_position;
	bool short_position;

	Strategy(double r, double t[2], double l) //int m,
	{
		//multiplicator = m;
		interest = r;
		trigger[0] = t[0];
		trigger[1] = t[1];
		//score[0] = s[0];
		//score[1] = s[1];
		loss = l;
		long_position = false;
		short_position = false;
		//tick = tick;
	}

	Transaction trade(double score[2], Factors::tickData tick)
	{
		if (long_position == false)
		{
			if (score[0]>trigger[0] && score[0]>score[1])  //
			{
				long_position = true;
				longRecord.openTime = tick.time;
				longRecord.openPrice = tick.lastPrice;
			}
		}
		else if (long_position == true)
		{
			if (score[1]>trigger[1] && score[1]>score[0])  // ?
			{
				long_position = false;
				longRecord.closeTime = tick.time;
				longRecord.closePrice = tick.bidPrice;
				longRecord.direction = "long - opppsite ";
				longRecord.profit = (longRecord.closePrice - longRecord.openPrice) -
					(longRecord.openPrice + longRecord.closePrice)*interest;
				longRecord.yield = longRecord.profit / longRecord.openPrice;
				short_position = true;
				shortRecord.openTime = tick.time;
				shortRecord.openPrice = tick.lastPrice;
				return longRecord;
			}
			else if (tick.bidPrice - longRecord.openPrice <= -loss)  //
			{
				long_position = false;
				longRecord.closeTime = tick.time;
				longRecord.closePrice = tick.bidPrice;
				longRecord.direction = "long - loss-cut";
				longRecord.profit = (longRecord.closePrice - longRecord.openPrice) -
					(longRecord.openPrice + longRecord.closePrice)*interest;
				longRecord.yield = longRecord.profit / longRecord.openPrice;
				return longRecord;
			}
			else if (tick.time.find("14:59") < 100) // 
			{
				long_position = false;
				longRecord.closeTime = tick.time;
				longRecord.closePrice = tick.bidPrice;
				longRecord.direction = "long - transaction closed";
				longRecord.profit = (longRecord.closePrice - longRecord.openPrice) -
					(longRecord.openPrice + longRecord.closePrice)*interest;
				longRecord.yield = longRecord.profit / longRecord.openPrice;
				return longRecord;
			}
		}

		if (short_position == false)
		{
			if (score[1]>trigger[1] && score[1]>score[0])  //
			{
				short_position = true;
				shortRecord.openTime = tick.time;
				shortRecord.openPrice = tick.lastPrice;
			}
		}
		else if (short_position == true)
		{
			if (score[0]>trigger[0] && score[0]>score[1])  //
			{
				short_position = false;
				shortRecord.closeTime = tick.time;
				shortRecord.closePrice = tick.askPrice;
				shortRecord.direction = "short - opposite";
				shortRecord.profit = (shortRecord.openPrice - shortRecord.closePrice) -
					(shortRecord.openPrice + shortRecord.closePrice)*interest;
				shortRecord.yield = shortRecord.profit / shortRecord.openPrice;
				long_position = true;
				longRecord.openTime = tick.time;
				longRecord.openPrice = tick.lastPrice;
				return shortRecord;
			}
			else if (shortRecord.openPrice - tick.askPrice <= -loss)
			{
				short_position = false;
				shortRecord.closeTime = tick.time;
				shortRecord.closePrice = tick.askPrice;
				shortRecord.direction = "short - loss-cut";
				shortRecord.profit = (shortRecord.openPrice - shortRecord.closePrice) -
					(shortRecord.openPrice + shortRecord.closePrice)*interest;
				shortRecord.yield = shortRecord.profit / shortRecord.openPrice;
				return shortRecord;
			}
			else if (tick.time.find("14:59") < 100)
			{
				short_position = false;
				shortRecord.closeTime = tick.time;
				shortRecord.closePrice = tick.askPrice;
				shortRecord.direction = "short - transaction closed";
				shortRecord.profit = (shortRecord.openPrice - shortRecord.closePrice) -
					(shortRecord.openPrice + shortRecord.closePrice)*interest;
				shortRecord.yield = shortRecord.profit / shortRecord.openPrice;
				return shortRecord;
			}
		}
		return emptyRecord;
	}
private:
};

int main()
{
	initParallel();
	setNbThreads(4);
	/*int n = nbThreads();
	cout << n << endl;*/
	time_t start, end;
	double tcost;
	time(&start);
	const int FACTORNUM = 6;
	const int TICKNUM = 10;
	const int FACTORSIZE = FACTORNUM * TICKNUM;
	//string pre = "../../data/ru0001_";
	string file = "2014630";
	//string app = ".csv";
	string filename = "../../data/ru201406m/ru0001_" + file + ".csv";
	ifstream infile(filename);
	if (!infile) {
		cout << "file not exsit";
		exit(1);
	}

	/* import neuralnet's weights */
	string weightFiles[] = { "../../model/6_factors/weight_0.csv" ,"../../model/6_factors/weight_2.csv" ,"../../model/6_factors/weight_4.csv" };
	string biasFiles[] = { "../../model/6_factors/weight_1.csv" ,"../../model/6_factors/weight_3.csv" ,"../../model/6_factors/weight_5.csv" };
	time_t start1 = clock();
	NeuralNet net = NeuralNet(weightFiles, biasFiles);
	time_t end1 = clock();
	cout << "matrixIO" << double(end1 - start1) / CLOCKS_PER_SEC << endl;

	//cout << net.weights.weight0.block(0,0,4,3) << endl;
	//cout << "*****************************" << endl;
	//cout << net.weights.weight1.block(0, 0, 4, 3) << endl;
	//cout << "#################################" << endl;
	//cout << net.weights.weight2.block(0, 0, 4, 3) << endl;

	/*  import max and min value for normalize  */
	string maxfile = "../../model/6_factors/input_max.csv";
	vector<vector<double>> input_max;
	input_max = readCSV(maxfile);
	string minfile = "../../model/6_factors/input_min.csv";
	vector<vector<double>> input_min;
	input_min = readCSV(minfile);

	string csvstr;
	deque<double> factorlist;
	//bool long_position = false;
	//bool short_position = false;
	double trigger[2] = { 0.977, 0.974 };
	double loss = 40.0;
	double interest = 0.5e-4;
	Strategy strategy = Strategy(interest, trigger, loss);
	//vector<ArrayXXf> prediction;
	vector<Strategy::Transaction>tradeDetails;
	//Strategy::Transaction record;
	/*string tmp;
	getline(infile, tmp, '\n');*/
	while (getline(infile, csvstr, '\n'))
	{
		//tickData tick;
		//tick = strToTick(csvstr);
		//Factors *fct;
		//fct = new Factors(csvstr);
		Factors fct = Factors(csvstr);
		for (int i = 0; i < FACTORNUM; ++i)
		{
			factorlist.push_back(fct.factor[i]);
		}
		if (factorlist.size() == FACTORSIZE)
		{
			/* input data nomalization */

			RowVectorXf input(FACTORSIZE);
			for (int i = 0; i < FACTORSIZE; i++)
			{
				double n_input;
				n_input = (factorlist[i] - input_min[i][0]) / (input_max[i][0] - input_min[i][0]);
				input(i) = n_input;
			}
			/* prediction */
			//time_t start2 = clock();
			net.prediction = net.predict(input);
			//time_t end2 = clock();
			//cout << "matrix" << double(end2 - start2) / CLOCKS_PER_SEC << endl;
			
			//prediction.push_back(net.prediction);
			/*cout << net.prediction << endl;
			cout << net.prediction(1) << " " << net.prediction(2) << endl;*/
			/* trade */
			double score[2] = { net.prediction(1), net.prediction(2) };
			Strategy::Transaction record = strategy.trade(score, fct.tick);
			if (!record.direction.empty())
			{
				tradeDetails.push_back(record);
			}
			/* delete the first tick */
			for (int i = 0; i < FACTORNUM; i++)
			{
				factorlist.pop_front();
			}
		}
	}

	/* save the trading details */
	cout << tradeDetails.size() << endl;
	string tmp1 = "../../trade/tradeRecord_" + file + ".csv";
	//string tmp2 = "../../trade/tradProfits_" + file + ".csv";
	ofstream tradeRecord(tmp1);
	//ofstream tradeProfit(tmp2);
	tradeRecord << "Direction,OpenTime,OpenPrice,CloseTime,ClosePrice,Profit,Yield" << endl;
	//tradeProfit << "Direction,Profits,Returns" << endl;
	//double accumulateProfit = 0.0;
	//double accumulateReturn = 0.0;
	for (int i = 0; i < tradeDetails.size(); i++)
	{
		tradeRecord << tradeDetails[i].direction << "," << tradeDetails[i].openTime << "," << tradeDetails[i].openPrice <<
			"," << tradeDetails[i].closeTime << "," << tradeDetails[i].closePrice << "," << tradeDetails[i].profit << ","
			<< tradeDetails[i].yield << endl;
		//accumulateProfit += tradeDetails[i].profit;
		//accumulateReturn += tradeDetails[i].yield;
		//tradeProfit << tradeDetails[i].direction << "," << accumulateProfit << "," << accumulateReturn << endl;
	}

	//cout << prediction.size() << endl;
	/*for (int i = 0; i < prediction.size();i++)
	{
	cout << prediction[i] << endl;
	}*/
	/*ofstream outfile("../../prediction/predict.csv");
	for (int i = 0; i < prediction.size(); i++)
	{
	outfile << prediction[i](0) << "," << prediction[i](1) << "," << prediction[i](2) << endl;
	}*/
	time(&end);
	tcost = difftime(end, start);
	cout << tcost << endl;

	return 0;
}
