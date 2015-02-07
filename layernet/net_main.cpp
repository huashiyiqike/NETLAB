#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <climits>
#include <glog/logging.h>
#include <iostream>
#include "nnet/nnet.h"
#include "leveldb/db.h"
#include "../layernet.pb.h"
#include "io/data.h"
#include "utils/io.h"

using namespace layernet;
using std::string;
using std::cout;
using std::endl;
using std::vector;
void initlogging(char* a,const Task& task){
	FLAGS_colorlogtostderr = true;
	FLAGS_log_dir = task.trainer().snapto_db();
	FLAGS_minloglevel = task.loglevel();
	if(!task.log() || task.checkgradient() || task.sample())
		FLAGS_logtostderr = true; // just to std
	else google::InitGoogleLogging(a);
}
int main(int argc,char *argv[]){

	if(argc < 2) LOG(FATAL)<< "You need to specify the configuration file.";
	cout.precision(15);
//	cout.setf(std::ios::unitbuf);
		Task task;
		CHECK(ReadProtoFromTextFile(argv[1],&task));
		if(argc==5) task.set_device(atoi(argv[4]));
		leveldb::Options options;
		options.create_if_missing = true;
		leveldb::DB * db;
		leveldb::Status status = leveldb::DB::Open(options,task.trainer().snapto_db(),&db);
		CHECK(status.ok()) << "Failed to open leveldb " << task.trainer().snapto_db();
		delete db;
		initlogging(argv[0],task);

		int tmp1=0;
		if(task.datap().name()=="t_ball_label") tmp1=1;
		task.set_tmax(task.datap().t()-task.datap().duration()-tmp1);

		INetTrainer *net_trainer;
		if(task.gpu()) {
			LOG(INFO) << "GPU starting";
			mshadow::InitTensorEngine(task.device());
			net_trainer=CreateNet("gpu",task);
		}
		else net_trainer=CreateNet("cpu",task);
		net_trainer->InitModel();

		string tmp,all;
		std::ifstream input(argv[1]);
		while(std::getline(input,tmp))
		all += tmp + '\n';
		input.close();
		LOG(ERROR) << all.c_str();

		//	int start_counter=0;
		unsigned long elapsed = 0;
		time_t start = time( NULL );

		IIterator<DataBatch>* itr_valid=NULL,* itr_test=NULL;

		if(task.valid()) {
			itr_valid=CreateIterator(task.datap_valid());
			itr_valid->Init();
			itr_valid->BeforeFirst();
		}
		if(task.test()) {
			CHECK(argc>=4)<<"Need to specify test range!";
			itr_test=CreateIterator(task.datap_test());
			itr_test->Init();
			itr_test->BeforeFirst();
			double err;
			if(task.err_in_train()) {
				err=net_trainer->AIS( itr_test, atoi(argv[2]) ,atoi(argv[3]) );
			}
			else {
				err=net_trainer->Evaluate(itr_test);
			}
			LOG(ERROR)<<"test error is "<<err;
			elapsed = (unsigned long)(time(NULL) - start);
			LOG(ERROR)<<elapsed<<"sec elapsed";
			exit(0);
		}
		IIterator<DataBatch>* itr_train=CreateIterator(task.datap());
		itr_train->Init();
		itr_train->BeforeFirst();
		if(task.save_interval()== 0 ) {
			task.set_save_interval(task.trainer().print_interval());
		}
		if(!task.sample()) {
			while(net_trainer->Round() <= task.trainer().num_round()) {
				while( itr_train->Next() ) {
					if(task.checkgradient()) {
						net_trainer->GradientCheck( itr_train->Value() );
						exit(0);
					}
					net_trainer->Update( itr_train->Value());
					if( net_trainer->Round() % task.trainer().print_interval()== 0 ) {
						elapsed = (unsigned long)(time(NULL) - start);
						LOG(ERROR)<<elapsed<<"sec elapsed";
						fflush( stdout );
					}

					if( net_trainer->Round() % task.save_interval()== 0 ) {
						if(task.trainer().save())
						net_trainer->SaveModel();
					}

					if(itr_valid&&net_trainer->Round() %task.trainer().num_trainround()==0) {
						double err=net_trainer->Evaluate( itr_valid );
						LOG(ERROR)<<"valid error is "<<err;
					}

				}
			}
		}
		else {
			net_trainer->Sample(task);
		}
		LOG(INFO)<<"End of program!";
	}
