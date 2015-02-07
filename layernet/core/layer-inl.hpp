#include "core.h"
#include "op.h"
#include "../utils/io.h"
#include "../compile_condition.h"

namespace layernet {
	using namespace mshadow;
	using namespace mshadow::expr;
	using namespace mshadow::utils;
	using google::protobuf::Message;

	template< typename xpu >
	struct peepConnec {
			ILayer<xpu>* from;
			ILayer<xpu>* to;
			//pointers to continuous memory
			mshadow::Tensor<xpu,1> w;
			mshadow::Tensor<xpu,1> dw;
			/** all connection have allocated memory for bias, not using is ok */
			peepConnec(ILayer<xpu>* fromlayer,ILayer<xpu> *tolayer) :
					from(fromlayer), to(tolayer){
				w.dptr = Wone<xpu>::instance().ptr;
				dw.dptr = Wone<xpu>::instance().dptr;
				int tmp;
				if(to->info.num_in() == 0) {
					tmp = from->info.num_out();
					w.shape[0] = to->info.num_out();
				}
				else {
					tmp = from->info.num_out();
					w.shape[0] = to->info.num_in();
				}
				w.shape.stride_ = w.shape[0];
				dw.shape = w.shape;
				Wone<xpu>::instance().ptr += tmp;
				Wone<xpu>::instance().dptr += tmp;
				Wone<xpu>::instance().weight_size += tmp;

				LOG(ERROR)<< from->info.name() << from->info.alias() << "->"
				<< to->info.name() << to->info.alias() << " weights:"
				<< w.shape[1] << "-->" << w.shape[0] << " total:" << tmp
				<< " weight_size " << Wone<xpu>::instance().weight_size;

			}
			// no bias
			inline void Forward_t(int t, int lag = 0) {
				// peep is direct connected, repmat for samples
				to->out.mat(t) += from->out.mat(t - lag)
				* repmat(w, to->out.mat(t).shape[1]);
			}
			inline void Backprop_t(int t, int lag = 0) {
				from->in.mat(t) += to->in.mat(t + lag)
				* repmat(w, from->in.mat(t + lag).shape[1]);
				dw += sumall_except_dim<0>(from->out.mat(t) * to->in.mat(t + lag));
			}
		};
	template< typename xpu >
	struct InitValue {
			mshadow::Tensor<xpu,1> b;
			mshadow::Tensor<xpu,1> db;
			InitValue(int size){
				b.dptr = Wone<xpu>::instance().ptr;
				db.dptr = Wone<xpu>::instance().dptr;
				b.shape[0] = size;
				b.shape.stride_ = size;
				db.shape = b.shape;
				Wone<xpu>::instance().ptr += size;
				Wone<xpu>::instance().dptr += size;
				Wone<xpu>::instance().weight_size += size;
			}
	};

	template< typename xpu >
	struct Connec {
			ILayer<xpu>* from;
			ILayer<xpu>* to;
			// for drawing weight, put into parameters.m
			//pointers to continuous memory
			mshadow::Tensor<xpu,2> w;
			mshadow::Tensor<xpu,2> dw;
			/** all connection have allocated memory for bias, not using is ok */
			mshadow::Tensor<xpu,1> b;
			mshadow::Tensor<xpu,1> db;
			mshadow::Tensor<xpu,1> vb;
			mshadow::Tensor<xpu,1> dvb;
			mshadow::Tensor<xpu,2> randh;
			mshadow::Tensor<xpu,2> randv;
			bool bias,delay;
			Connec(ILayer<xpu>* fromlayer,ILayer<xpu> *tolayer,bool delay =
					false,bool bias = true) :
					from(fromlayer), to(tolayer), bias(bias), delay(delay){
				if(fromlayer->info.name() == tolayer->info.name()) {
					if(fromlayer->info.alias() == "")
						CHECK(delay == true)
														<< "self connection should be delayed:"
														<< fromlayer->info.name();
				}
				w.dptr = Wone<xpu>::instance().ptr;
				dw.dptr = Wone<xpu>::instance().dptr;
				int tmp;
				if(to->info.num_in() == 0) {
					tmp = from->info.num_out() * to->info.num_out();
					w.shape[0] = to->info.num_out();
				}
				else {
					tmp = from->info.num_out() * to->info.num_in();
					w.shape[0] = to->info.num_in();
				}
				w.shape[1] = from->info.num_out();
				w.shape.stride_ = w.shape[0];
				dw.shape = w.shape;
				Wone<xpu>::instance().ptr += tmp;
				Wone<xpu>::instance().dptr += tmp;
				Wone<xpu>::instance().weight_size += tmp;

				int tmp2 = 0;
				if(bias) {
					b.dptr = Wone<xpu>::instance().ptr;
					db.dptr = Wone<xpu>::instance().dptr;
					if(to->info.num_in() == 0) {
						tmp2 = to->info.num_out();
						b.shape[0] = to->info.num_out();
					}
					else {
						tmp2 = to->info.num_in();
						b.shape[0] = to->info.num_in();
					}
					b.shape.stride_ = w.shape[0];
					db.shape = b.shape;

					Wone<xpu>::instance().ptr += tmp2;
					Wone<xpu>::instance().dptr += tmp2;
					Wone<xpu>::instance().weight_size += tmp2;
				}
				//rbm V_bias
				int tmp3 = 0;
				if(fromlayer->info.type() != LayerProto_LayerType_RBM
						&& tolayer->info.type() == LayerProto_LayerType_RBM
						&& !delay) {
					randv = NewTensor<xpu>(
							Shape2(from->out.data.shape[1],
									from->out.data.shape[0]),0.0);
					randh = NewTensor<xpu>(
							Shape2(to->out.data.shape[1],to->out.data.shape[0]),
							0.0);

					vb.dptr = Wone<xpu>::instance().ptr;
					dvb.dptr = Wone<xpu>::instance().dptr;
					tmp3 = from->info.num_out();
					vb.shape[0] = from->info.num_out();

					vb.shape.stride_ = w.shape[0];
					dvb.shape = vb.shape;

					Wone<xpu>::instance().ptr += tmp3;
					Wone<xpu>::instance().dptr += tmp3;
					Wone<xpu>::instance().weight_size += tmp3;
				}

#ifdef DEBUG
				b = 0.01;
#endif

				LOG(ERROR)<< from->info.name() << from->info.alias() << "->"
				<< to->info.name() << to->info.alias() << " weights:"
				<< w.shape[1] << "*" << w.shape[0] << " total:"
				<< tmp + tmp2 + tmp3 << " weight_size "
				<< Wone<xpu>::instance().weight_size;
			}
			//this exist because to->out.mat(t) may not be zero at first, B is added to bh.
			inline void Forward_t_inter(int t, int lag = 0) {
				to->B_.mat(t) += dot(from->out.mat(t - lag), w);
				if (bias) {
					to->B_.mat(t) += repmat(b, to->out.mat(t - lag).shape[1]); //consider restore this for optimize
				}
			}

			inline void Forward_t(int t, int lag = 0) {
				to->out.mat(t) += dot(from->out.mat(t - lag), w);
				if (bias) {
					to->out.mat(t) += repmat(b, to->out.mat(t - lag).shape[1]); //consider restore this for optimize
				}
			}
			inline void Backprop_t(int t, int lag = 0) {
				from->in.mat(t) += dot(to->in.mat(t + lag), w.T());
				dw += dot(from->out.mat(t).T(), to->in.mat(t + lag));
				if (bias)
				db += sum_rows(to->in.mat(t + lag));
			}
			inline double AIS(int t, Tensor<xpu, 1> biasplus,
					mshadow::Random<xpu> &rnd_, double* beta, int numruns = 100) {
				int numcases = numruns;
				int numdims = w.shape[1];
				int numhids = w.shape[0];

				int numtest = 1;
				CHECK(from->out.mat(t).shape[1] == 1) << "just for one sample a time";
				static Tensor<xpu, 2> pd = NewTensor<xpu>(Shape2(numtest, 1), 0.0);
				static Tensor<cpu, 2> pd_tmp_numtest = NewTensor<cpu>(
						Shape2(numtest, 1), 0.0);
				static Tensor<xpu, 2> out = NewTensor<xpu>(Shape2(numtest, numhids),
						0.0);
				static Tensor<xpu, 1> tmpcase2 = NewTensor<xpu>(Shape1(numtest), 0.0);
				static Tensor<xpu, 2> visbiases_base = NewTensor<xpu>(
						Shape2(1, numdims), 0.0), randvtmp = NewTensor<xpu>(
						Shape2(numcases, numdims), 0.0), randhtmp = NewTensor<xpu>(
						Shape2(numcases, numhids), 0.0), visbias_base = NewTensor<xpu>(
						Shape2(numruns, numdims), 0.0), hidbias = NewTensor<xpu>(
						Shape2(numruns, numhids), 0.0), visbias = NewTensor<xpu>(
						Shape2(numruns, numdims), 0.0), visbiases = NewTensor<xpu>(
						Shape2(1, numdims), 0.0), hidbiases = NewTensor<xpu>(
						Shape2(1, numhids), 0.0), logww = NewTensor<xpu>(
						Shape2(numcases, 1), 0.0), negdata = NewTensor<xpu>(
						Shape2(numcases, numdims), 0.0), Wh = NewTensor<xpu>(
						Shape2(numcases, numhids), 0.0), Bv_base = NewTensor<xpu>(
						Shape2(numcases, 1), 0.0), Bv = NewTensor<xpu>(
						Shape2(numcases, 1), 0.0), expWh = NewTensor<xpu>(
						Shape2(numcases, numhids), 0.0), poshidprobs = NewTensor<xpu>(
						Shape2(numcases, numhids), 0.0), poshidstates = NewTensor<xpu>(
						Shape2(numcases, numhids), 0.0);
				static Tensor<cpu, 2> tmp_numcases = NewTensor<cpu>(Shape2(numcases, 1),
						0.0f), tmp_numdims = NewTensor<cpu>(Shape2(1, numdims), 0.0f);

				static Tensor<xpu, 1> tmpcase = NewTensor<xpu>(Shape1(numcases), 0.0),
				biasplusin = NewTensor<xpu>(Shape1(numhids), 0.0);
				biasplusin = biasplus + b;

				logww = 0.0;
				visbiases_base = 0.0;

				hidbias = repmat(biasplusin, numcases);
				visbias = repmat(vb, numcases);
				visbiases = broadcast<0>(vb, visbiases.shape);
				hidbiases = broadcast<0>(biasplusin, hidbiases.shape);

				rnd_.SampleUniform(randvtmp);
				negdata = F<op::samples>(F<op::sigmoid>(visbias_base), randvtmp);
				logww -= dot(negdata, visbiases_base.T());
				logww -= numhids * log(2);

				Wh = dot(negdata, w);
				Wh += hidbias;

				Bv_base = dot(negdata, visbiases_base.T());
				Bv = dot(negdata, visbiases.T());

				double bb;
				for (int i = 0; i < 14492; i++) {
					bb = beta[i];
					expWh = F<op::Exp>(bb * Wh);
					logww += (1 - bb) * Bv_base + bb * Bv;
					tmpcase = sumall_except_dim<1>(F<op::Log1>(expWh));
					logww += broadcast<1>(tmpcase, logww.shape);

					poshidprobs = expWh / (1.0 + expWh);
					rnd_.SampleUniform(randhtmp);
					poshidstates = F<op::samples>(poshidprobs, randhtmp);

					randvtmp = dot(poshidstates, w.T());
					negdata = (1 - bb) * visbias_base + bb * (randvtmp + visbias);
					rnd_.SampleUniform(randvtmp);
					negdata = F<op::samples>(F<op::sigmoid>(negdata), randvtmp);

					Wh = dot(negdata, w);
					Wh += hidbias;
					Bv_base = dot(negdata, visbiases_base.T());
					Bv = dot(negdata, visbiases.T());

					expWh = F<op::Exp>(bb * Wh);
					logww -= (1 - bb) * Bv_base + bb * Bv;
					tmpcase = sumall_except_dim<1>(F<op::Log1>(expWh));
					logww -= broadcast<1>(tmpcase, logww.shape);
				}
				expWh = F<op::Exp>(Wh);
				logww += dot(negdata, visbiases.T());

				tmpcase = sumall_except_dim<1>(F<op::Log1>(expWh));
				logww += broadcast<1>(tmpcase, logww.shape);

				Copy(tmp_numcases, logww);
				double maxs = -10000;
				for (int i = 0; i < numcases; i++) {
					if (tmp_numcases[i][0] > maxs)
					maxs = tmp_numcases[i][0];
				}

				double alpha = maxs - log(FLT_MAX) / 2;
				logww -= alpha;
				logww = F<op::Exp>(logww);
				Copy(tmp_numcases, logww);
				double sums = 0.0;
				for (int i = 0; i < numcases; i++) {
					sums += tmp_numcases[i][0];
				}
				double r_AIS = log(sums) + alpha - log(numcases);

				visbiases_base = F<op::Log1>(F<op::Exp>(visbiases_base));
				Copy(tmp_numdims, visbiases_base);
				sums = 0.0;
				for (int i = 0; i < numdims; i++) {
					sums += tmp_numdims[0][i];
				}
				double logZZ_base = sums + (numhids) * log(2);
				double logZ = r_AIS + logZZ_base;

				pd = dot(from->out.mat(t), visbiases.T());

				out = dot(from->out.mat(t), w);
				out += repmat(biasplusin, numtest);	//consider restore this for optimize
				tmpcase2 = sumall_except_dim<1>(F<op::Log1exp>(out));
				pd += broadcast<1>(tmpcase2, pd.shape);
				double logprob = 0.0;
				Copy(pd_tmp_numtest, pd);
				for (int i = 0; i < numtest; i++)
				logprob += pd_tmp_numtest[i][0];
				logprob /= numtest;
				std::cout<<"t:"<<t<<std::endl;
				std::cout<<"logprob:"<<logprob<<std::endl;
				std::cout<<"logZ:"<<logZ<<std::endl;
				return logprob - logZ;
			}

			inline void Sample_t(Tensor<xpu, 2> H, Tensor<xpu, 2> V,
					Random<xpu> &rnd_, int t, int steps, bool visgauss = false) {
#ifdef DEBUG
						V=0.01f;
#else
						rnd_.SampleUniform(V);
#endif
						for (int i = 0; i < to->info.cd_n(); i++) {
							LOG(INFO) << "t " << t << "i " << i;
							H = dot(V, w);
							H += repmat(b, to->out.data.shape[1]) + to->B_.mat(t);
							H = F<op::sigmoid>(H);
							rnd_.SampleUniform(randh);
#ifndef DEBUG
						H = F<op::samples>(H, randh);
#endif
						V = dot(H, w.T());
						V += repmat(vb, to->out.data.shape[1]);
#ifdef DEBUG
						V=F<op::sigmoid>(V);
#else
						if (visgauss) {
							rnd_.SampleGaussian(randv);
							V += randv;
						}
						else {
							rnd_.SampleUniform(randv);
							V = F<op::samples>(F<op::sigmoid>(V), randv);//		from->in.mat(t)=1.0f;
						}
#endif
					}
					H = dot(V, w);
					H += repmat(b, to->out.data.shape[1]) + to->B_.mat(t);
					H = F<op::sigmoid>(H);
					V = dot(H, w.T());
					V += repmat(vb, to->out.data.shape[1]);
					if (!visgauss) {
						rnd_.SampleUniform(randv);
						V = F<op::samples>(F<op::sigmoid>(V),randv);
					}
					Copy(from->out.mat(t), V);
				}

				inline double grad_cd_t(bool propgrad, int t ,int cd_n,Tensor<xpu, 2> H, Tensor<xpu, 2> V,
						Random<xpu> &rnd_, int duration, bool visgauss = false) {
					//out should not change now;
					dw += dot(from->out.mat(t).T(), to->in.mat(t));
					db += sum_rows(to->in.mat(t));
					Copy(V, from->out.mat(t));
					H = dot(V, w);
					H += repmat(b, to->out.data.shape[1]) + to->B_.mat(t);
					H = F<op::sigmoid>(H);

					dw += dot(V.T(), H);
					dvb += sumall_except_dim<0>(V);
					to->in.mat(t) += H;
					db += sumall_except_dim<0>(H);
					for (int i = 0; i < cd_n; i++) {
						rnd_.SampleUniform(randh);
#ifndef DEBUG
						H = F<op::samples>(H, randh);
#endif

						V = dot(H, w.T());
						V += repmat(vb, to->out.data.shape[1]);
#ifdef DEBUG
						V=F<op::sigmoid>(V);
#else
						if (visgauss) {
							rnd_.SampleGaussian(randv);
							V += randv;
						}
						else {
							rnd_.SampleUniform(randv);
							V = F<op::samples>(F<op::sigmoid>(V), randv);
						}

#endif
						H = dot(V, w);
						H += repmat(b, to->out.data.shape[1]) + to->B_.mat(t);
						H = F<op::sigmoid>(H);
					}
					dw -= dot(V.T(), H);
					dvb -= sumall_except_dim<0>(V);
					db -= sumall_except_dim<0>(H);
					to->in.mat(t) -= H;

					double err = 0.0;
					if (propgrad) {
						static TensorContainer<cpu, 1> tmpv(Shape1(from->in.data.shape[0]));
						V = dot(H, w.T());
						V += repmat(vb, to->out.data.shape[1]);
						if (!visgauss) {
							V = F<op::sigmoid>(V);
						}
						V -= from->out.mat(t);
						V *= V;
						randv[0] = sumall_except_dim<0>(F<op::Abs>(V));
						Copy(tmpv, randv[0]);
						int start = 0;
						if (duration > 0) {
							int len = tmpv.shape[0] / duration;
							start = 2 * len;
						}
						for (index_t i = start; i < tmpv.shape[0]; i++)
						err += tmpv[i];
					}
					return err;
				}
			};

	template< typename xpu >
	class RBMLayer : public ILayer<xpu> {
		public:
			mshadow::Random<xpu> &rnd;
			using ILayer<xpu>::in;
			using ILayer<xpu>::out;
			using ILayer<xpu>::info;
			using ILayer<xpu>::B_;
			std::vector<Connec<xpu> > connection;
			Node<xpu> H;
			double beta[14502];
			std::vector<Node<xpu> > V;
			InitValue<xpu> *binit;
			NodeFactory<xpu> nfactory;
			int duration;

		public:
			RBMLayer(const LayerProto &l,mshadow::Random<xpu> &r,Node<xpu> &in,
					Node<xpu> &out) :
					ILayer<xpu>(l,in,out), rnd(r){
			}
			virtual void InitLayer(std::map<std::string,ILayer<xpu>*> & map){
				int index = 0;
				double tmp = 0.0;
				while(tmp <= 0.5) {
					beta[index++] = tmp;
					tmp += 0.001;
				}
				tmp = 0.5;
				while(tmp <= 0.9) {
					beta[index++] = tmp;
					tmp += 0.0001;
				}
				tmp = 0.9;
				while(tmp <= 1) {
					beta[index++] = tmp;
					tmp += 0.00001;
				}
				connection.reserve(20);
				V.reserve(10);
				duration = map["input"]->info.duration();
				for(int i = 0 ; i < ILayer<xpu>::info.from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.from(i)],this,
									false));
					V.push_back(nfactory.CreateNode());
					V.back().data.shape =
							map[ILayer<xpu>::info.from(i)]->out.data.shape;
				}
				for(int i = 0 ; i < ILayer<xpu>::info.delay_from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.delay_from(i)],
									this,true));
				}
				B_ = nfactory.CreateNode();
				B_.data.shape = out.data.shape;
				H = nfactory.CreateNode();
				H.data.shape = out.data.shape;
				InitNodes();
				if(info.traininit()) {
					binit = new InitValue<xpu>(in.data.shape[0]);
					LOG(ERROR)<< "binit " << " total:" << in.data.shape[0]
					<< " weight_size " << Wone<xpu>::instance().weight_size;
				}
			}
			virtual void InitModel(void) {
				//	DLOG(INFO)<<"sigmoid layer initilzing";
			}
			virtual double CalcError(bool prop_grad, int t = 0) {
				return 0.0f;
			}
			virtual double Forward_AIS(bool is_train, int t = 0) {
				double prob = 0.0;
				if(t==0&&info.traininit()) {
					B_.mat(t)= repmat(binit->b, out.data.shape[1]);
				}
				//input self connection etc, not just one connection, but many connections.
				for ( int i = 0; i <(int) connection.size(); i++) {
					if (!connection[i].delay) {
						connection[i].Forward_t(t);
					}
					else {
						if (t > 0) { // for memorization of H bias for grad_cd backward
							/** self connection one step delay*/
							connection[i].Forward_t_inter(t, 1); //add to out.mat
						}
					}
				}
				out.mat(t) += B_.mat(t);
				out.mat(t) = F<op::sigmoid>(out.mat(t));
				for (unsigned int i = 0; i < connection.size(); i++) {
					if (!connection[i].delay) { // just RBM
						prob = connection[i].AIS(t, B_.mat(t)[0], rnd, beta);
					}
				}
				return prob;
			}
			virtual void Forward(bool is_train, int t = 0) {
				if(t==0&&info.traininit()) {
					B_.mat(t)= repmat(binit->b, out.data.shape[1]);
				}
				//input self connection etc, not just one connection, but many connections.
				for ( int i = 0; i <(int) connection.size(); i++) {

					if (!connection[i].delay) {
						if(!is_train) {
							connection[i].Sample_t(H.mat(t), V[i].mat(t),
									rnd, t, ILayer<xpu>::info.cd_n(),
									info.visgauss());
						}
						connection[i].Forward_t(t);
					}
					else {
						if (t > 0) { // for memorization of H bias for grad_cd backward
							/** self connection one step delay*/
							connection[i].Forward_t_inter(t, 1); //add to out.mat
						}
					}
				}
				out.mat(t) += B_.mat(t);
				out.mat(t) = F<op::sigmoid>(out.mat(t));

			}

			virtual double Backprop_CD(bool prop_grad, int t = 0, int cd_n = 5,int length=0) {
				double inerror = 0.0;
				in.mat(t) *= F<op::sigmoid_grad>(out.mat(t)); //dB[t] = F_t * (1 - h_[t]) * h_[t];  for bptt
				for (unsigned int i = 0; i < connection.size(); i++) {
					/**self connection do not propagate*/
					if (!connection[i].delay) {
						//this is not bp, but in.mat(t) is calculated with db
						inerror += connection[i].grad_cd_t(prop_grad, t, cd_n, H.mat(t), V[i].mat(t), rnd, duration,
								info.visgauss());
					}
					else {
						if (t > 0)
						connection[i].Backprop_t(t - 1, 1);
					}
				}
				if (t == 0&&info.traininit()) {
					binit->db = sumall_except_dim<0>(in.mat(0));
					binit->db*=length;
				}
				return inerror;
			}

			virtual void Selfgrad(bool prop_grad) {
			}
			virtual void SaveModel(mshadow::utils::IStream &fo) const {
			}
			virtual void LoadModel(mshadow::utils::IStream &fi) {
			}

			virtual ~RBMLayer(void) {
			}
			virtual void Forwardinit() {
				B_.data = 0.0;
				out.data = 0.0;
			}
			virtual void Backpropinit() {
				in.data = 0.0;
			}

			protected:
			inline void InitNodes(void) {
				size_t i = 0;
				mshadow::Shape<4> s;
				for (; i < V.size(); ++i) {
					s = V[i].data.shape;
					V[i].Pin();
					V[i].Unpin();
					V[i].data = 0.0;
					printf("RBM node[%d].shape: %u,%u,%u,%u\n", (int) i, s[3], s[2],
							s[1], s[0]);
				}
				B_.Pin();
				B_.Unpin();
				B_.data = 0.0;
				s = B_.data.shape;
				printf("RBM node[%d].shape: %u,%u,%u,%u\n", (int) i++, s[3], s[2], s[1],
						s[0]);
				H.Pin();
				H.Unpin();
				H.data = 0.0;
				s = H.data.shape;
				printf("RBM node[%d].shape: %u,%u,%u,%u\n", (int) i++, s[3], s[2], s[1],
						s[0]);
			}
		}
		;

	template< typename xpu >
	class LSTMLayer : public ILayer<xpu> {
		public:
			mshadow::Random<xpu> &rnd;
			using ILayer<xpu>::in;
			using ILayer<xpu>::out;
			using ILayer<xpu>::info;
			std::vector<Node<xpu> > nodes;
			std::vector<Connec<xpu>*> in_inG;
			std::vector<Connec<xpu>*> in_forG;
			std::vector<Connec<xpu>*> in_outG;
			std::vector<Connec<xpu>*> in_preinG;
			InitValue<xpu> *binit;
			peepConnec<xpu> *peep_inG,*peep_forG,*peep_outG;
			Connec<xpu> *cell_preoutG;
			NodeFactory<xpu> nfactory;
			ILayer<xpu> *inG,*forG,*outG;
			ILayer<xpu> *preinG,*preoutG,*state;
			LayerProto lproto;
		public:
			LSTMLayer(const LayerProto &l,mshadow::Random<xpu> &r,Node<xpu> &in,
					Node<xpu> &out) :
					ILayer<xpu>(l,in,out), rnd(r), lproto(l){
				lproto.set_num_out(l.num_in());
				nodes.reserve(300);
				int T = in.data.shape[2];
				int samples = in.data.shape[1];

				for(int i = 0 ; i < 3 * 2 ; i++) {
					nodes.push_back(nfactory.CreateNode());
					nodes.back().data.shape = Shape4(1,T,samples,l.num_in());
				}
				/**   num_in gates in num_in blocks*/
				int nodeindex = -1;
				lproto.set_num_in(l.num_in());
				lproto.set_num_out(l.num_in());
				lproto.set_alias("_inG");
				inG = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));
				lproto.set_alias("_forG");
				forG = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));
				lproto.set_alias("_outG");
				outG = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));

				for(int j = 0 ; j < 3 * 2 ; j++) {
					nodes.push_back(nfactory.CreateNode());
					nodes.back().data.shape = Shape4(1,T,samples,l.num_out());
				}
				lproto.set_num_in(l.num_out());
				lproto.set_num_out(l.num_out());
				lproto.set_alias("_preinG");
				preinG = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));
				lproto.set_alias("_preoutG");
				preoutG = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));
				lproto.set_alias("_state");
				state = new ILayer<xpu>(lproto,nodes.at(++nodeindex),
						nodes.at(++nodeindex));
				/** restore value*/
				lproto.set_num_in(l.num_in());
				lproto.set_num_out(l.num_out());

				if(lproto.num_in() == 0) lproto.set_num_in(l.num_out());
				lproto.set_num_cell(lproto.num_out() / lproto.num_in());
				CHECK(lproto.num_cell() == 1);
				this->InitNodes();
			}

			virtual void InitLayer(std::map<std::string,ILayer<xpu>*> & map){
				bool bias;
				for(int i = 0 ; i < lproto.from_size() ; i++) {
					bias = true;
					in_inG.push_back(
							new Connec<xpu>(map[ILayer<xpu>::info.from(i)],inG,
									false,bias));
					in_inG.back()->w -= 2.0;
					in_forG.push_back(
							new Connec<xpu>(map[ILayer<xpu>::info.from(i)],forG,
									false,bias));
					in_outG.push_back(
							new Connec<xpu>(map[ILayer<xpu>::info.from(i)],outG,
									false,bias));
					in_preinG.push_back(
							new Connec<xpu>(map[ILayer<xpu>::info.from(i)],
									preinG,false,bias));
				}
				for(int i = 0 ; i < lproto.delay_from_size() ; i++) {
					bias = true;
					in_inG.push_back(
							new Connec<xpu>(
									map[ILayer<xpu>::info.delay_from(i)],inG,
									true,bias));
					in_inG.back()->w -= 2.0;
					in_forG.push_back(
							new Connec<xpu>(
									map[ILayer<xpu>::info.delay_from(i)],forG,
									true,bias));
					in_outG.push_back(
							new Connec<xpu>(
									map[ILayer<xpu>::info.delay_from(i)],outG,
									true,bias));
					in_preinG.push_back(
							new Connec<xpu>(
									map[ILayer<xpu>::info.delay_from(i)],preinG,
									true,bias));
				}
				/** inner connections */
				peep_inG = new peepConnec<xpu>(state,inG);
				//		peep_inG->w-=1.0;
				peep_forG = new peepConnec<xpu>(state,forG);
				peep_outG = new peepConnec<xpu>(state,outG);
				//	cell_preoutG=new Connec<xpu>(state,preoutG,true);
				if(info.traininit()) {
					binit = new InitValue<xpu>(in.data.shape[0]);
					LOG(ERROR)<< "binit " << " total:" << in.data.shape[0]
					<< " weight_size " << Wone<xpu>::instance().weight_size;
				}

			}
			virtual void InitModel(void) {
				//	DLOG(INFO)<<"LSTM layer initilzing";
			}

			virtual void Forward(bool is_train,int t = 0) {
				//input self connection etc, not just one connection, but many connections.
				for(int i = 0;
						i < lproto.from_size() + lproto.delay_from_size();
						i++) {
					/**self connection do not propagate*/
					if(!in_inG[i]->delay) {
						in_inG[i]->Forward_t(t);
						in_forG[i]->Forward_t(t);
						in_outG[i]->Forward_t(t);
						in_preinG[i]->Forward_t(t);
					}
					/** self connection one step delay*/
					else if(t > 0) {
						in_inG[i]->Forward_t(t,1);
						in_forG[i]->Forward_t(t,1);
						in_outG[i]->Forward_t(t,1);
						in_preinG[i]->Forward_t(t,1);
					}
				}
				/** ingate forgate state update*/
				if(t > 0) { // peep,here delay
					peep_inG->Forward_t(t,1);
					peep_forG->Forward_t(t,1);
				}
				inG->out.mat(t) = F<op::sigmoidn>(inG->out.mat(t));
				forG->out.mat(t) = F<op::sigmoidn>(forG->out.mat(t));
				/** cell state update*/
				preinG->out.mat(t) = F<op::tanh>(preinG->out.mat(t));

				state->out.mat(t) = preinG->out.mat(t) * inG->out.mat(t);
				if(t > 0) {
					state->out.mat(t) += forG->out.mat(t)
					* state->out.mat(t - 1);
				}

				peep_outG->Forward_t(t); // peep, here not delay,state is new
				outG->out.mat(t) = F<op::sigmoidn>(outG->out.mat(t));
				preoutG->out.mat(t) = F<op::tanh>(state->out.mat(t));

				out.mat(t) = outG->out.mat(t) * preoutG->out.mat(t);
				if(t==0&&info.traininit()) {
					out.mat(t) += repmat(binit->b, out.data.shape[1]);
				}
			}
			void spy(std::string a,int t) {}

			virtual void Backprop(bool prop_grad, int t = 0,int length=0) {
				outG->in.mat(t) = in.mat(t) * preoutG->out.mat(t)
				* F<op::sigmoidn_grad>(outG->out.mat(t));
				/**cell err derivatives dE/dState*/ //plus because peep error there
				state->in.mat(t) +=in.mat(t)*F<op::tanh_grad>(preoutG->out.mat(t))
				* outG->out.mat(t);
				peep_outG->Backprop_t(t);
				if(t>0) {
					state->in.mat(t-1) += state->in.mat(t ) * forG->out.mat(t );
				}

				//cell error
				preinG->in.mat(t) = state->in.mat(t) * inG->out.mat(t)
				* F<op::tanh_grad>(preinG->out.mat(t));

				if (t > 0) {
					forG->in.mat(t) = F<op::sigmoidn_grad>(forG->out.mat(t))
					* state->in.mat(t) * state->out.mat(t - 1);
				}
				inG->in.mat(t) = F<op::sigmoidn_grad>(inG->out.mat(t))
				* state->in.mat(t) * preinG->out.mat(t);
				if(t>0) {
					/**  peep bp need error now, not propagate further, so it is t */
					peep_forG->Backprop_t(t-1, 1);
					peep_inG->Backprop_t(t-1, 1);
				}
				for (int i = 0; i < lproto.from_size() + lproto.delay_from_size();
						i++) {
					/**self connection do not propagate*/
					if (!in_inG[i]->delay) {
						// bp to lower layer
						in_inG[i]->Backprop_t(t, 0);
						in_forG[i]->Backprop_t(t, 0);
						in_outG[i]->Backprop_t(t, 0);
						in_preinG[i]->Backprop_t(t, 0);
					}
					/** self connection back prop one step delay*/
					else if (t > 0) {
						in_inG[i]->Backprop_t(t - 1, 1);
						in_forG[i]->Backprop_t(t - 1, 1);
						in_outG[i]->Backprop_t(t - 1, 1);
						in_preinG[i]->Backprop_t(t - 1, 1);
					}
				}

				if (t == 0&&info.traininit()) {
					binit->db = sumall_except_dim<0>(in.mat(0));
					binit->db*=length;
				}
			}
			virtual void SaveModel(mshadow::utils::IStream &fo) const {
			}
			virtual void LoadModel(mshadow::utils::IStream &fi) {
			}

			virtual ~LSTMLayer(void) {
			}

			virtual void Forwardinit() {
				out.data = 0.0;
				inG->out.data = 0.0;
				forG->out.data = 0.0;
				outG->out.data = 0.0;
				preoutG->out.data = 0.0;
				preinG->out.data = 0.0;
				state->out.data = 0.0;
			}
			virtual void Backpropinit() {
				in.data = 0.0;
				state->in.data = 0.0;
			}
			protected:
			inline void InitNodes(void) {
				for (size_t i = 0; i < nodes.size(); ++i) {
					mshadow::Shape<4> s = nodes[i].data.shape;
					nodes[i].Pin();
					nodes[i].Unpin();
					nodes[i].data = 0.0;
					printf("LSTM node[%d].shape: %u,%u,%u,%u\n", (int) i, s[3], s[2],
							s[1], s[0]);
				}
			}

		};

	template< typename xpu >
	class InputLayer : public ILayer<xpu> {
		public:
			InputLayer(const LayerProto &l,mshadow::Random<xpu> &r,
					Node<xpu> &in,Node<xpu> &out) :
					ILayer<xpu>(l,in,out), rnd(r){
				//	DLOG(INFO)<<"input layer initialized";
			}
			virtual void InitLayer(std::map<std::string,ILayer<xpu>*> & map){
			}
			virtual void InitModel(void){
				//DLOG(INFO)<<"input layer initilzing";
			}
			virtual void SaveModel(mshadow::utils::IStream &fo) const{
			}
			virtual void LoadModel(mshadow::utils::IStream &fi){
			}

		public:
			mshadow::Random<xpu> &rnd;
			using ILayer<xpu>::in;
			using ILayer<xpu>::out;
			std::vector<Connec<xpu> > connection;

			virtual ~InputLayer(void){
			}
			virtual void Forward(bool is_train,int t = 0){
			}
			virtual void Backprop(bool prop_grad,int t = 0,int length = 0){

			}
			virtual void Forwardinit(){
				LOG(FATAL)<<"No init data here!";
			}
			virtual void Backpropinit() {
			}
			protected:
		}
		;

	template< typename xpu >
	class SigmoidLayer : public ILayer<xpu> {
		public:
			SigmoidLayer(const LayerProto &l,mshadow::Random<xpu> &r,
					Node<xpu> &in,Node<xpu> &out) :
					ILayer<xpu>(l,in,out), rnd(r){
			}
			virtual void InitLayer(std::map<std::string,ILayer<xpu>*> & map){
				connection.reserve(20);
				for(int i = 0 ; i < ILayer<xpu>::info.from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.from(i)],this,
									false));
				}
				for(int i = 0 ; i < ILayer<xpu>::info.delay_from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.delay_from(i)],
									this,true));
				}
				if(info.traininit()) {
					binit = new InitValue<xpu>(in.data.shape[0]);
					LOG(ERROR)<< "binit " << " total:" << in.data.shape[0]
					<< " weight_size " << Wone<xpu>::instance().weight_size;
				}
			}
			virtual void InitModel(void) {
				//DLOG(INFO)<<"full layer initilzing";
			}
			virtual void SaveModel(mshadow::utils::IStream &fo) const {
			}
			virtual void LoadModel(mshadow::utils::IStream &fi) {
			}
			virtual void Selfgrad(bool prop_grad) {
			}
			virtual ~SigmoidLayer() {
			}
			virtual void Forwardinit() {
				out.data = 0.0;
			}
			virtual void Backpropinit() {
				in.data = 0.0;
			}
			public:
			mshadow::Random<xpu> &rnd;
			using ILayer<xpu>::in;
			using ILayer<xpu>::out;
			using ILayer<xpu>::info;
			InitValue<xpu> *binit;
			std::vector<Connec<xpu> > connection;

			virtual void Forward(bool is_train,int t = 0) {
				//input self connection etc, not just one connection, but many connections.
				for(unsigned int i = 0; i < connection.size(); i++) {
					/**self connection do not propagate*/
					if(!connection[i].delay) {
						connection[i].Forward_t(t);
					}
					/** self connection one step delay*/
					else if(t > 0) {
						connection[i].Forward_t(t,1);
					}
				}
				if(t==0&&info.traininit()) {
					out.mat(t) += repmat(binit->b, out.data.shape[1]);
				}
				out.mat(t)=F<op::sigmoid>(out.mat(t));
			}

			virtual void Backprop(bool prop_grad,int t = 0,int length = 0) {
				in.mat(t)*=F<op::sigmoid_grad>(out.mat(t));
				for(unsigned int i = 0; i < connection.size(); i++) {
					/**self connection do not propagate*/
					if(!connection[i].delay) {
						// bp to lower layer
						connection[i].Backprop_t(t);
					}
					/** self connection back prop one step delay*/
					else if(t > 0) {
						connection[i].Backprop_t(t - 1,1);
					}
				}
				if (t == 0&&info.traininit()) {
					binit->db = sumall_except_dim<0>(in.mat(0));
					binit->db*=length;
				}
			}
			protected:
		}
		;

	template< typename xpu >
	class FullConnectLayer : public ILayer<xpu> {
		public:
			FullConnectLayer(const LayerProto &l,mshadow::Random<xpu> &r,
					Node<xpu> &in,Node<xpu> &out) :
					ILayer<xpu>(l,in,out), rnd(r){
			}
			virtual void InitLayer(std::map<std::string,ILayer<xpu>*> & map){
				connection.reserve(20);
				for(int i = 0 ; i < ILayer<xpu>::info.from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.from(i)],this,
									false));
				}
				for(int i = 0 ; i < ILayer<xpu>::info.delay_from_size() ; i++) {
					connection.push_back(
							Connec<xpu>(map[ILayer<xpu>::info.delay_from(i)],
									this,true));
				}
				if(info.traininit()) {
					binit = new InitValue<xpu>(in.data.shape[0]);
					LOG(ERROR)<< "binit " << " total:" << in.data.shape[0]
					<< " weight_size " << Wone<xpu>::instance().weight_size;
				}
			}
			virtual void InitModel(void) {
				//DLOG(INFO)<<"full layer initilzing";
			}
			virtual void SaveModel(mshadow::utils::IStream &fo) const {
			}
			virtual void LoadModel(mshadow::utils::IStream &fi) {
			}
			virtual void Selfgrad(bool prop_grad) {
			}
			virtual ~FullConnectLayer() {
			}
			virtual void Forwardinit() {
				out.data = 0.0;
			}
			virtual void Backpropinit() {
				in.data = 0.0;
			}
			public:
			mshadow::Random<xpu> &rnd;
			using ILayer<xpu>::in;
			using ILayer<xpu>::out;
			using ILayer<xpu>::info;
			InitValue<xpu> *binit;
			std::vector<Connec<xpu> > connection;

			virtual void Forward(bool is_train,int t = 0) {
				//input self connection etc, not just one connection, but many connections.
				for(unsigned int i = 0; i < connection.size(); i++) {
					/**self connection do not propagate*/
					if(!connection[i].delay) {
						connection[i].Forward_t(t);
					}
					/** self connection one step delay*/
					else if(t > 0) {
						connection[i].Forward_t(t,1);
					}
				}
				if(t==0&&info.traininit()) {
					out.mat(t) += repmat(binit->b, out.data.shape[1]);
				}
			}
			virtual void Backprop(bool prop_grad,int t = 0,int length = 0) {
				for(unsigned int i = 0; i < connection.size(); i++) {
					/**self connection do not propagate*/
					if(!connection[i].delay) {
						// bp to lower layer
						connection[i].Backprop_t(t);
					}
					/** self connection back prop one step delay*/
					else if(t > 0) {
						connection[i].Backprop_t(t - 1,1);
					}
				}
				if (t == 0&&info.traininit()) {
					binit->db = sumall_except_dim<0>(in.mat(0));
					binit->db*=length;
				}
			}
			protected:
		}
		;
	template< typename xpu >
	inline ILayer<xpu>* CreateLayer(const LayerProto &l,
			mshadow::Random<xpu> &rnd,Node<xpu> &in,Node<xpu> &out){
		switch(l.type()){
			case LayerProto_LayerType_DATA:
				return new InputLayer<xpu>(l,rnd,in,out);
			case LayerProto_LayerType_RBM:
				return new RBMLayer<xpu>(l,rnd,in,out);
			case LayerProto_LayerType_LSTM:
				return new LSTMLayer<xpu>(l,rnd,in,out);
			case LayerProto_LayerType_FULL_CONNECT:
				return new FullConnectLayer<xpu>(l,rnd,in,out);
			case LayerProto_LayerType_SIGMOID:
				return new SigmoidLayer<xpu>(l,rnd,in,out);

			default:
				Error("unknown layer type");
		}
		return NULL;

	}

}
;
