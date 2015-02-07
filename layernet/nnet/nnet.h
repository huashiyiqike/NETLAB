#ifndef NNET_H
#define NNET_H
#pragma once
/*!
 * \file cxxnet_nnet.h
 * \brief trainer abstraction
 * \author Bing Xu, Tianqi Chen
 */
#include <vector>
#include <glog/logging.h>
#include "../../layernet.pb.h"
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"
#include "../io/data.h"

namespace layernet {
	/*! \brief interface for network */
	class INetTrainer {
		public:
			virtual ~INetTrainer(void){
			}
			/*! \brief set model parameters, call this before everything, including load model */
			//	virtual void SetParam(const char *name,const char *val) = 0;
			/*! \brief random initalize model */
			virtual void InitModel(void) = 0;
			/*! \brief save model to stream */
			virtual void SaveModel(bool wait=false)  = 0; //mshadow::utils::IStream &fo
			/*! \brief load model from stream */
			virtual void LoadModel() = 0; //mshadow::utils::IStream &fi
			/*!
			 * \brief inform the updater that a new round has been started
			 * \param round round counter
			 */
			//		virtual void StartRound(int round) = 0;
			/*!
			 * \brief update model parameter
			 * \param training data batch
			 */
			virtual void Update(const DataBatch& data,bool train=true) = 0;
			virtual void Sample(Task task)=0;
			virtual int& Round()=0;
			virtual void GradientCheck(const DataBatch& data) = 0;
//        /*! \brief  evaluate a test statistics, output results into fo */
			virtual double Evaluate(IIterator<DataBatch> *iter_eval) = 0;
			virtual double AIS(IIterator<DataBatch> *iter_eval,int index,int end) = 0;
//        /*! \brief  predict labels */
//        virtual void Predict( std::vector<float> &preds, const DataBatch& batch ) = 0;
	};
}
;

namespace layernet {
	/*!
	 * \brief create a CPU net implementation
	 * \param net_type network type, used to select trainer variants
	 */
	INetTrainer* CreateNetCPU(Task task,int net_type = 0);
	/*!
	 * \brief create a GPU net implementation
	 * \param net_type network type, used to select trainer variants
	 * \param devid device id
	 */
	INetTrainer* CreateNetGPU(Task task,int net_type = 0);
	/*!
	 * \brief create a net implementation
	 * \param net_type network type, used to select trainer variants
	 * \param device device type
	 */
	inline INetTrainer* CreateNet(const char *device,Task task){
		if(!strcmp(device,"cpu")) return CreateNetCPU(task);
		if(!strcmp(device,"gpu")) return CreateNetGPU(task);
		LOG(FATAL)<< "unknown device type";
		return NULL;
	}
}
;
#endif // CXXNET_H
