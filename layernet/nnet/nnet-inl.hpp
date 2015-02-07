#ifndef NET_INL_HPP
#define NET_INL_HPP
#pragma once
#include <map>
#include <fstream>
#include "../core/core.h"
#include "nnet.h"
#include "leveldb/db.h"
#include "../utils/io.h"
#include "../compile_condition.h"
#include "../utils/thread.h"
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <omp.h>

namespace layernet {
	using namespace mshadow::utils;
	using namespace mshadow::expr;

	template< typename xpu >
	class NeuralNet {
		public:
			std::vector<Node<xpu> > nodes;
			/*! \brief layers in the neural net */
			std::vector<ILayer<xpu>*> layers;
			std::map<std::string,ILayer<xpu>*> layer_map;
			/*! \brief random number generator */
			mshadow::Random<xpu> rnd;
			/*! \brief node factory */
			NodeFactory<xpu> nfactory;
		public:
			NeuralNet(void) :
					rnd(0){
				LOG(ERROR)<< "net initializing";
			}
			/*! \brief destructor */
			virtual ~NeuralNet(void) {
				this->FreeSpace();
			}
			/*! \brief input node */
			inline Node<xpu>& in(void) {
				return nodes[1];
			}
			/*! \brief output node */
			inline Node<xpu>& out(void) {
				return nodes.back();
			}
			inline Node<xpu>& out_before(void) {
				return nodes.at(nodes.size() - 2);
			}
			/*! \brief intialize model parameters */
			void InitModel(Task task) {
				this->FreeSpace();
				layers.reserve(100);
				nodes.reserve(200);
				for (int i = 0; i < task.layers_size(); i++) {
					Assert(layers.size() == (size_t) i);
					Assert(layer_map.size() == (size_t) i);
					/** with LSTM cells, need change the push for nodes, only the first and end of LSTM nodes be pushed?*/
					nodes.push_back(nfactory.CreateNode());

					/** if specify input size*/
					if (task.layers(i).num_in() != 0)
					nodes.back().data.shape = Shape4(1, task.tmax(),
							task.datap().batchsize(), task.layers(i).num_in());
					else
					nodes.back().data.shape = Shape4(1, task.tmax(),
							task.datap().batchsize(), task.layers(i).num_out());
					nodes.push_back(nfactory.CreateNode());
					nodes.back().data.shape = Shape4(1, task.tmax(),
							task.datap().batchsize(), task.layers(i).num_out());
					layers.push_back(
							CreateLayer(task.layers(i), rnd, nodes.at(nodes.size() - 2),
									nodes.back()));
					layer_map.insert(
							std::make_pair(layers.back()->info.name(), layers.back()));
					printf("total node number: %lu\n", nodes.size());
				} /** LSTM is different for having many nodes*/

				printf("total node number: %lu\n", nodes.size());
				this->InitNodes();
				for (real_t i = 0; i < layers.size(); i++)
				layers[i]->InitLayer(layer_map);
				Wone<xpu>::instance().Weight.shape[0] =
				Wone<xpu>::instance().weight_size;
				Wone<xpu>::instance().dWeight.shape[0] =
				Wone<xpu>::instance().weight_size;
				Wone<xpu>::instance().dWeight_momentum.shape[0] =
				Wone<xpu>::instance().weight_size;
				Wone<xpu>::instance().dWeight_1.shape[0] =
				Wone<xpu>::instance().weight_size;
				Wone<xpu>::instance().dWeight_2.shape[0] =
				Wone<xpu>::instance().weight_size;
			}

			/*! \brief save model to file */
			inline void SaveModel(const int round, const std::string &filename) const {
				static TensorContainer<cpu, 1> tmp(Wone<xpu>::instance().Weight.shape);
				static TensorContainer<cpu, 1> tmp2(Wone<xpu>::instance().Weight.shape);
				char tmpchar[100];
				sprintf(tmpchar, "/round%d_weight.txt", round);
				std::string tmpstr = filename;
				tmpstr += tmpchar;
				std::ofstream outw(tmpstr.c_str(), std::ofstream::out);
				Copy(tmp, Wone<xpu>::instance().Weight);
				std::cout << Wone<xpu>::instance().Weight.shape[0] << std::endl;
				outw << round << std::endl;
				outw << Wone<xpu>::instance().Weight.shape[0] << std::endl;
				for (unsigned int i = 0; i < Wone<xpu>::instance().Weight.shape[0];
						i++) {
					outw << tmp[i] << std::endl;
				}
				std::cout << "write weight into " << tmpstr << std::endl;
			}

			/*! \brief load model from stream */
			inline void LoadModel(int &round, const TrainerProto &trainerp) {
				std::string filename = trainerp.snapfrom_db();
				static TensorContainer<cpu, 1> tmp(Wone<xpu>::instance().Weight.shape);
				static TensorContainer<cpu, 1> tmp2(Wone<xpu>::instance().Weight.shape);
				std::ifstream inw(filename.c_str(), std::ifstream::in);
				CHECK(inw != NULL) << filename.c_str() << " open fails";
				inw >> round;
				inw >> Wone<xpu>::instance().Weight.shape[0];
				for (unsigned int i = 0; i < Wone<xpu>::instance().Weight.shape[0];
						i++) {
					CHECK(inw >> tmp[i]) << i << "th weight load failed!";
				}
				Copy(Wone<xpu>::instance().Weight, tmp);
				std::cout << "load weight from " << filename << std::endl;
			}

			inline double CalcError() {
				layers[layers.size() - 1]->in.resetNode();
				return layers[layers.size() - 1]->CalcError(true);
			}

//			/*!
//			 * \brief forward prop
//			 * \param is_train whether is training phase
//			 */
			inline void Forward(bool is_train, int length) {
				for (size_t i = 1; i < layers.size(); ++i) {
					layers[i]->Forwardinit();
				}

				for (int t = 0; t < length; t++)
				for (size_t i = 0; i < layers.size(); ++i) {
					layers[i]->Forward(is_train, t);
				}
			}

			inline double Forward_AIS(bool is_train, int length) {
				double err = 0, inerr = 0;
				for (size_t i = 1; i < layers.size(); ++i) {
					layers[i]->Forwardinit();
				}

				for (int t = 0; t < length; t++)
				for (size_t i = 0; i < layers.size(); ++i) {
					if (layers[i]->info.type() != LayerProto_LayerType_RBM)
					layers[i]->Forward(is_train, t);
					else {
						inerr = layers[i]->Forward_AIS(is_train, t);
						inerr = inerr < 0 ? inerr : 0;
						err += inerr;
						LOG(ERROR) << "t:" << (t + 1) << " LL:" << inerr
						<< " average:" << err / (t + 1);
					}
				}

				return err / (double) length;
			}

			/*! \brief backprop */

			inline double Backprop(int length, int round = 0, int interval = 0,
					bool err_in_train = false) {
				double err = 0.0;
				int cd;
				int tmp;
				if (err_in_train) {
					tmp = 0;
				}
				else {
					tmp = 1;
				} // last layer in.mat may have been calculated before bp.
				for (size_t i = 0; i < layers.size() - tmp; ++i) {
					layers[i]->Backpropinit();
				}
				/**just the middle of net reset*/

				/**do not backprop input*/
				for (int t = length - 1; t >= 0; t--) {
					for (int i = (int) layers.size() - 1; i >= 0; --i) {
						if (layers[i]->info.type() != LayerProto_LayerType_RBM)
						layers[i]->Backprop(i != 1, t);
						else {
							if (round < 100)
							cd = 5;
							else if (round < 1100)
							cd = 10;
							else if (round < 50000)
							cd = 25;
							else if (round < 100000)
							cd = 35;
							else
							cd =50;
							err += layers[i]->Backprop_CD(round % interval == 0, t, cd,
									length);
						}
					}
				}
				Wone<xpu>::instance().dWeight /= length * nodes[0].data.shape[1];
				return err / length / nodes[0].data.shape[1];
			}

//			/*! \brief update model parameters  */
			inline void Update(Task &task, int round) {
				double lr = task.learnrate();
				if (task.learnrate() == 1)
				lr = lr * (1 - round / task.trainer().num_round());
				Wone<xpu>::instance().dWeight = F<op::bound>(
						Wone<xpu>::instance().dWeight);
				int pos=1;
				if (!task.plus_gradient()) {
					pos=-1;
				}
				Wone<xpu>::instance().dWeight/=task.update_period();
				switch(task.learnmethod()) {
					case Task_LearnMethod_SGD: {
						Wone<xpu>::instance().dWeight_momentum = task.momentum()
						* Wone<xpu>::instance().dWeight_momentum
						+ pos*lr * Wone<xpu>::instance().dWeight;
						Wone<xpu>::instance().Weight +=
						Wone<xpu>::instance().dWeight_momentum;
						Wone<xpu>::instance().dWeight = 0;
						break;
					}
					case Task_LearnMethod_RPROP: {
						Wone<xpu>::instance().dWeight_1=Wone<xpu>::instance().dWeight_1*0.95+0.05*Wone<xpu>::instance().dWeight;
						Wone<xpu>::instance().dWeight_2=Wone<xpu>::instance().dWeight_2*0.95+
						0.05*Wone<xpu>::instance().dWeight_2*Wone<xpu>::instance().dWeight_2;
						Wone<xpu>::instance().dWeight_momentum = task.momentum()
						* Wone<xpu>::instance().dWeight_momentum
						+ lr*pos* Wone<xpu>::instance().dWeight/F<op::power>((Wone<xpu>::instance().dWeight_2-
										Wone<xpu>::instance().dWeight_1*Wone<xpu>::instance().dWeight_1+0.0001),1/2
						);
						Wone<xpu>::instance().Weight +=
						Wone<xpu>::instance().dWeight_momentum;
						Wone<xpu>::instance().dWeight = 0;
						break;
					}
					case Task_LearnMethod_NONE_DEFINE: {
						Wone<xpu>::instance().dWeight_momentum = task.momentum()
						* Wone<xpu>::instance().dWeight_momentum
						+ pos*lr * Wone<xpu>::instance().dWeight;
						Wone<xpu>::instance().Weight +=
						Wone<xpu>::instance().dWeight_momentum;
						Wone<xpu>::instance().dWeight = 0;
						break;
					}
				}
			}

			private:
			/*! \brief check the node shapes */
			inline void InitNodes(void) {
				for (size_t i = 0; i < nodes.size(); ++i) {
					mshadow::Shape<4> s = nodes[i].data.shape;
					nodes[i].Pin();
					nodes[i].Unpin();
					printf("node[%d].shape: %u,%u,%u,%u\n", (int) i, s[3], s[2], s[1],
							s[0]);
				}
			}
			/*! \brief set parameters */
			inline void FreeSpace(void) {
				for (size_t i = 0; i < nodes.size(); ++i) {
					nodes[i].FreeSpace();
				}
				for (size_t i = 0; i < layers.size(); ++i) {
					delete layers[i];
				}
				nodes.clear();
				layers.clear();
			}
		}
		;

		/*! \brief implementation of neural network trainer */
	template< typename xpu >
	class NetTrainer : public INetTrainer {
		public:
			NetTrainer(){
			}
			NetTrainer(Task &task) :
					task(task){
				loss_type = 0;
				round = 0;
				update_period = 1;
				sample_counter = 0;
				eval_train = 1;
				savelock.Init(1);
			}
			virtual ~NetTrainer(void){
			}
			virtual int& Round(){
				return round;
			}
			void weight_rand(Tensor<xpu,1> Weight){
				Weight *= 1.1259e+15;
				Weight = mshadow::expr::F<op::fmode>(Weight,2000.0);
				Weight -= 1000.0;
				Weight *= 0.0001;
			}
			virtual void InitModel(void){
				/** use proto to allocate weight memory*/
				Wone<xpu>::instance();
				net.InitModel(task);
				if(task.trainer().load()) LoadModel();

			}

			inline static CXXNET_THREAD_PREFIX Saveinner(void* nettrainer) {
				static_cast<NetTrainer<xpu>* >(nettrainer)->net.SaveModel( static_cast<NetTrainer<xpu>* >(nettrainer)->round,
						static_cast<NetTrainer<xpu>* >(nettrainer)->task.trainer().snapto_db());
				utils::ThreadExit( NULL );
				return NULL;
			}

			virtual void SaveModel(bool wait=false) {
				savelock.Wait();
				save_thread.Start(Saveinner,this);
				savelock.Post();
				if(wait)
				save_thread.Join();
			}
			virtual void LoadModel() {
				net.LoadModel(round,task.trainer());
			}

			virtual double AIS(IIterator<DataBatch> *iter_eval,int index,int end) {
				err = 0.0;
				int samples = 0;
				for(int i =1;i<index;i++) { // index for the indexth sample for calculation
					CHECK(iter_eval->Next());
				}
				for(int i=index;i<=end;i++) {
					CHECK(iter_eval->Next());
					const DataBatch& batch = iter_eval->Value();
					err += this->AISTemp(batch);
					samples++;
					LOG(ERROR)<<"till file"<<i<<"logp:"<<(err/samples);
				}
				err /= samples;
				return err;
			}

			virtual double Evaluate(IIterator<DataBatch> *iter_eval) {
				err = 0.0;
				int inst = 0;
				int samples = 0;
				while(iter_eval->Next()) {
					inst++;
					const DataBatch& batch = iter_eval->Value();
					samples = batch.data.shape[1];
					if(!task.err_in_train()) {
						this->PreparePredTemp(batch);
						if(batch.data.shape[2] == 1) {
							for(unsigned int i = 0; i < temp.shape[1]; i++) {
								index_t maxidx = 0;
								for(unsigned int j = 0; j < temp.shape[0];
										j++) {
									if(temp[i][j] > temp[i][maxidx]) maxidx = j;
								}
								err += (batch.labels[0][0][0][i] != maxidx);
								//    printf("%lf\n",err);
							}
						}
						else {
							if(!task.variable_length()) {
								for(size_t t = 0; t < tempT.shape[2]; t++) {
									for(size_t i = 0; i < tempT.shape[1]; i++) {
										if(batch.lengthlist[i] == t) {
											err +=
											(tempT[t][i][0]
													- batch.labels[0][t][i][0])
											* (tempT[t][i][0]
													- batch.labels[0][t][i][0]);
										}
									}
								}
							}
							else {
								double inerr=0.0;
								for(size_t t = 0; t < batch.lengthlist[0]; t++) {
									double p=1.0;
									for(size_t i = 0; i < tempT.shape[0]; i++) {
										if(batch.data[0][t][0][i]==0) {
											p*=(1-tempT[t][0][i]);
										}
										else {
											p*=tempT[t][0][i];
										}
									}
									inerr += log(p);
								}
								err+=inerr/batch.lengthlist[0];
							}

						}
					}
					else {
						err += this->ReconstructErrorTemp(batch);
					}
				}
				err /= samples * inst;
				return err;
			}

			double norm(mshadow::Tensor<cpu,1> tmp) {
				double tmpp = 0.0;
				for(unsigned int i = 0; i < tmp.shape[0]; i++) {
					tmpp += tmp[i] * tmp[i];
				}
				return sqrt(tmpp);
			}

			virtual void GradientCheck(const DataBatch& batch) {
				TensorContainer<cpu,1> dweight_compare(
						Wone<xpu>::instance().dWeight.shape);
				TensorContainer<cpu,1> Weight_backup(
						Wone<xpu>::instance().dWeight.shape);
				Copy(Weight_backup,Wone<xpu>::instance().Weight);
				double EPSILON = 0.0001;				// 10 ^ -4;
				for(unsigned int i = 0; i < dweight_compare.shape[0]; i++) {
					double err = 0,err2 = 0;
					Copy(Wone<xpu>::instance().Weight,Weight_backup);
					Wone<xpu>::instance().dWeight = 0.0;
					Wone<xpu>::instance().Weight[i] += EPSILON;
					this->PreparePredTemp(batch);
					if(batch.data.shape[2] == 1) {
						for(unsigned int i = 0; i < temp.shape[1]; i++) {
							index_t maxidx = 0;
							for(unsigned int j = 0; j < temp.shape[0]; j++) {
								if(temp[i][j] > temp[i][maxidx]) maxidx = j;
							}
							err += (batch.labels[0][0][0][i] != maxidx);
						}
					}
					else {
						for(size_t t = 0; t < tempT.shape[2]; t++) {
							for(size_t i = 0; i < tempT.shape[1]; i++) {
								if(batch.lengthlist[i]-1 == t) {
									err += (tempT[t][i][0]
											- batch.labels[0][t][i][0])
									* (tempT[t][i][0]
											- batch.labels[0][t][i][0]);
								}
							}
						}
					}

					Wone<xpu>::instance().Weight[i] -= 2 * EPSILON;
					this->PreparePredTemp(batch);
					if(batch.data.shape[2] == 1) {
						for(unsigned int i = 0; i < temp.shape[1]; i++) {
							index_t maxidx = 0;
							for(unsigned int j = 0; j < temp.shape[0]; j++) {
								if(temp[i][j] > temp[i][maxidx]) maxidx = j;
							}
							err2 += (batch.labels[0][0][0][i] != maxidx);
						}
					}
					else {
						for(size_t t = 0; t < tempT.shape[2]; t++) {
							for(size_t i = 0; i < tempT.shape[1]; i++) {
								if(batch.lengthlist[i]-1 == t) {
									err2 += (tempT[t][i][0]
											- batch.labels[0][t][i][0])
									* (tempT[t][i][0]
											- batch.labels[0][t][i][0]);
								}
							}
						}
					}
					dweight_compare[i] = (err - err2) / EPSILON / 4/batch.data.shape[2];
				}
				Copy(Wone<xpu>::instance().Weight,Weight_backup);
				Wone<xpu>::instance().dWeight = 0.0;
				this->PreparePredTemp(batch);
				if(batch.data.shape[2] != 1)
				this->SetLoss(batch.labels,batch.lengthlist);
				else this->SetLoss(batch.labels[0][0][0]);
				net.Backprop(batch.data.shape[2]);
				for(unsigned int i = 0; i < dweight_compare.shape[0]; i++) {
					LOG(ERROR)<<"w"<<i+1<<" "<< Wone<xpu>::instance().dWeight[i]<<" "<<" should be "<<dweight_compare[i];
				}
				Copy(Weight_backup,Wone<xpu>::instance().dWeight);
				Weight_backup -= dweight_compare;
				double diff = norm(Weight_backup);
				LOG(ERROR)<<diff;
				Weight_backup += 2 * dweight_compare;
				diff /= norm(Weight_backup);
				LOG(ERROR)<< "Norm of the difference between numerical and analytical gradient"<<diff<<"(should be < 1e-9)";

			}
			double norm_nosqrt(mshadow::Tensor<xpu,1> tmp) {
				static Tensor<cpu,1> tmpcpu=NewTensor<cpu>(tmp.shape,0.0);
				Copy(tmpcpu,tmp);
				double tmpp = 0.0;
				for(unsigned int i = 0; i < tmpcpu.shape[0]; i++) {
					tmpp += tmpcpu[i] * tmpcpu[i];
				}
				return tmpp;
			}
			virtual void Update(const DataBatch& batch,bool train=true) {
				net.in().Pin();
				mshadow::Copy(net.in().data,batch.data);
				net.in().Unpin();

				int length;
				if(task.variable_length()) length=batch.lengthlist[0];
				else length=batch.data.shape[2];
				net.Forward(true,length);
				this->SyncOuput();
//
				if(net.out().data.shape[2] == 1) {
					net.out().Pin();
					mshadow::Copy(net.out().data[0][0],temp);
					net.out().Unpin();
				}
//
				if(!task.err_in_train()) {
					if(length != 1)
					this->SetLoss(batch.labels,batch.lengthlist);
					else this->SetLoss(batch.labels[0][0][0]);
					net.out().data/=task.datap().t()*task.datap().batchsize();
					err/=task.datap().t()*task.datap().batchsize();
				}
				if(task.err_in_train())
				err=net.Backprop(length,round,task.trainer().print_interval() ,task.err_in_train() );
				else net.Backprop(length,round,task.trainer().print_interval() ,task.err_in_train() );
				if(this->Round()% update_period==0) {
					net.Update(task,round);
				}
				if(task.trainer().print_interval()!=0&&round%task.trainer().print_interval()==0) {
					double tmpw=sqrt(norm_nosqrt(Wone<xpu>::instance().Weight)/Wone<xpu>::instance().Weight.shape[0]);
					double tmpdw=sqrt(norm_nosqrt(Wone<xpu>::instance().dWeight_momentum)/Wone<xpu>::instance().Weight.shape[0]);
					CHECK(tmpw<5)<<"w error";
					CHECK(tmpdw<5)<<"dw error";
					LOG(ERROR) << "epoch " << round << " err "<< err << " |w| "
					<< tmpw
					<<"|dw|"<<tmpdw;
					CHECK(tmpw<5)<<"w error";
					CHECK(tmpdw<5)<<"dw error";
				}
				this->Round()++;
			}
			virtual void Sample(Task task) {
				CHECK(net.in().data.shape[1]==1)<<"Should generate one sample a time!";
				Tensor<cpu,4> tmp=NewTensor<cpu>(net.in().data.shape,0.0f);
				if(!task.sample_text()) {
					CHECK(task.sample()==1)<<"should only produce one sample";
					net.Forward(false,net.in().data.shape[2]);
					Copy(tmp,net.in().data);
					py.save(tmp,tmp.shape[2],(int)sqrt(tmp.shape[0]));
				}
				else {
					static TensorContainer<cpu,4> tmp(net.in().data.shape);
					std::string tmpstr=task.trainer().snapto_db();
					tmpstr+="/generate.txt";
					std::ofstream outw(tmpstr.c_str(),std::ofstream::out);
					outw << task.sample_num() << " "<< tmp.shape[2]<<" "<<tmp.shape[0]<<std::endl;
					for(int i=0;i<task.sample_num();i++) {
						outw<<tmp.shape[2]<<std::endl;
						net.Forward(false,net.in().data.shape[2]);
						Copy(tmp,net.in().data );
						for( int t = 0; t <(int)tmp.shape[2];
								t++) {
							for(unsigned int i = 0; i < (int)tmp.shape[0];
									i++) {
								outw << tmp[0][t][0][i] <<" ";
							}
							outw <<std::endl;
						}

					}

					std::cout<<"Successfully write sample data into generate.txt"<<std::endl;
				}
			}
			protected:
			virtual double AISTemp(const DataBatch& batch) {
				net.in().Pin();
				mshadow::Copy(net.in().data,batch.data);
				net.in().Unpin();

				int length;
				if(task.variable_length()) length=batch.lengthlist[0];
				else length=batch.data.shape[2];
				std::cout<<"length:"<<length<<std::endl;
				err=net.Forward_AIS(true,length);
				return err;
			}

			virtual double ReconstructErrorTemp(const DataBatch& batch) {
				net.in().Pin();
				mshadow::Copy(net.in().data,batch.data);
				net.in().Unpin();

				int length;
				if(task.variable_length()) length=batch.lengthlist[0];
				else length=batch.data.shape[2];
				net.Forward(true,length);
				this->SyncOuput();
				if(net.out().data.shape[2] == 1) {
					net.out().Pin();
					mshadow::Copy(net.out().data[0][0],temp);
					net.out().Unpin();
				}
				err=net.Backprop(length,round,1, task.err_in_train() );
				return err;
			}
			virtual void PreparePredTemp(const DataBatch& batch) {
				net.in().Pin();
				mshadow::Copy(net.in().data,batch.data);
				net.in().Unpin();
				net.Forward(true,net.in().data.shape[2]);
				this->SyncOuput();
			}

			private:
			inline void SyncOuput(void) {
				mshadow::Shape<4> oshape = net.out().data.shape;

				net.out().Pin();
				if(net.out().data.shape[2] != 1) {
					tempT.Resize(
							mshadow::Shape3(oshape[2],oshape[1],oshape[0]));
					mshadow::Copy(tempT,net.out().data[0]);
				}
				else {
					temp.Resize(mshadow::Shape2(oshape[1],oshape[0]));
					mshadow::Copy(temp,net.out().data[0][0]);

				}
				net.out().Unpin();
			}
			inline void SetLoss0(mshadow::Tensor<cpu,1> pred,float label) {
				switch(loss_type) {
					case 0: {
						index_t k = static_cast<index_t>(label);
						utils::Assert(k < pred.shape[0],
								"label exceed output bound");
						pred[k] -= 1.0;
						break;
					}
					case 1:
					pred[0] -= label;
					break;
					case 2:
					pred[0] = 1.0/ (1.0 + std::exp(-pred[0])) - label;
					break;
					default:
					Error("unknown loss type");
				}
			}
			/** pred dim : samples*output size, label dim: samples for gradient check,do not touch */
			inline double SetLoss_t(Tensor<cpu,2> pred,Tensor<cpu,2> label) {
				/** pred dim : samples*output size, label dim: samples */
				/** for each sample*/
				err=0.0;
				for(size_t i = 0; i < pred.shape[1]; i++) {
					for(size_t j=0; j<pred.shape[0];j++) {
						if(label[i][j]==0) {
							err-=log(1-pred[i][j]);
							pred[i][j]=-log(1-pred[i][j]);
						}
						else {
							err-=log(pred[i][j]);
							pred[i][j]=-log(pred[i][j]);
						}
					}

				}
				return err;

			}

			inline void SetLoss(mshadow::Tensor<cpu,1> labels) {
				if(loss_type == 1 || loss_type == 2) {
					Assert(temp.shape[0] == 1,
							"regression can only have 1 output size");
				}
				if(eval_train != 0) {
				}
				for(index_t i = 0; i < temp.shape[1]; ++i) {
					this->SetLoss0(temp[i],labels[i]);
				}

				net.out().Pin();
				mshadow::Copy(net.out().data[0][0],temp);
				net.out().Unpin();
			}

			inline void SetLoss(mshadow::Tensor<cpu,4> labels,
					mshadow::Tensor<cpu,1> lengthlist) {
				err = 0.0;
				if(loss_type == 1 || loss_type == 2) {
					Assert(tempT.shape[0] == 1,
							"regression can only have 1 output size");
				}
				if(eval_train != 0) {
				}

				for(int t = lengthlist[0] - 1; t >= 0; --t) {
					this->SetLoss_t(tempT[t],labels[0][t]);
				}

				net.out_before().Pin();
				mshadow::Copy(net.out_before().data[0],tempT);
				net.out_before().data *= 1.0
				/ (tempT.shape[1] * update_period);
				net.out_before().Unpin();
			}
			public:
			int round;
			protected:
			Python_helper py;
			Task task;
			/*! \brief current round */
			utils::Thread save_thread;
			utils::Semaphore savelock;

			/*! \brief loss function */
			int loss_type;
			/*! \brief update period */
			int update_period;
			/*! \brief sample counter */
			int sample_counter;
			double err;
			/*! \brief evaluator */
			/*! \brief temp space */
			mshadow::TensorContainer<cpu,3> tempT; // t numsample out dim1;
			mshadow::TensorContainer<cpu,2> temp;// t numsample out dim1;
			/*! \brief true net */
			NeuralNet<xpu> net;
			/*! \brief tmp stoage of top index */
			std::vector<index_t> tmp_index_;
			/*! \brief show train eval */
			int eval_train;
			/*! \brief evaluator for train */
		}
		;

	template< typename xpu >
	INetTrainer* CreateNet_(Task task,int net_type = 0){
		switch(net_type){
			case 0:
				return new NetTrainer<xpu>(task);
			default:
				Error("unknown net type");
		}
		return NULL;
	}

}
;

#endif
