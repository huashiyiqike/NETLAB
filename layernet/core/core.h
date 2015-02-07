#ifndef CORE_H
#define CORE_H
#pragma once


#include <vector>
#include <limits>
#include <queue>
#include <utility>
#include <string>
#include <iostream>
#include <cstdlib>
#include <glog/logging.h>
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"
#include "mshadow/tensor_container.h"
#include "../../layernet.pb.h"
#include "../compile_condition.h"
#include <typeinfo>
namespace layernet {

	typedef mshadow::cpu cpu;
	typedef mshadow::gpu gpu;
	typedef mshadow::index_t index_t;
	typedef mshadow::real_t real_t;

	template< typename xpu >
	struct Node;
	template< typename xpu >
	inline void spy(mshadow::Tensor<xpu,2> data){
		for(unsigned int i = 0 ; i < data.shape[1] ; i++) {
			for(unsigned int j = 0 ; j < data.shape[0] ; j++) {
				printf("%d %f ",i,data[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	template< typename xpu >
	class ILayer {
		public:
			LayerProto info;
			Node<xpu> &in;
			Node<xpu> &out;
			Node<xpu> B_;
		public:
			explicit ILayer<xpu>(const LayerProto &l,Node<xpu> &in_,
					Node<xpu> &out_) :
					info(l), in(in_), out(out_){
			}

			virtual ~ILayer(void){
			}

			// public interface
			virtual void Forward(bool is_train,int t = 0){
			}
			;
			virtual double Forward_AIS(bool is_train,int t = 0){
				return 2 << 10;
			}
			;

			virtual void Forwardinit(){
			}
			;

			virtual double Backprop_CD(bool prop_grad,int t = 0,int cd_n = 5,
					int length = 0){
				return 0;
			}
			virtual void Backprop(bool prop_grad,int t = 0,int length = 0){
			}
			;
			virtual void Backpropinit(){
			}
			;
			virtual double CalcError(bool prop_grad,int t = 0){
				return 0.0f;
			}
			;

			virtual void Selfgrad(bool prop_grad){
			}
			;

		public:
			// interface code that not needed to be implemented by all nodes

			virtual void InitLayer(std::map<std::string,ILayer*> & map){
			}

			virtual void InitModel(void){
			}

			virtual void SaveModel(mshadow::utils::IStream &fo) const{
			}

			virtual void LoadModel(mshadow::utils::IStream &fi){
			}
	};
	template< typename xpu >
	struct Wone {
			static Wone& instance(int size = 3000000){
				using namespace mshadow;
				static Wone weight_singleton_(size);
				return weight_singleton_;
			}
			/**this is just for initialization*/
			real_t *ptr;
			real_t *dptr;
			/**this is for update*/
			mshadow::Tensor<xpu,1> Weight;
			mshadow::Tensor<xpu,1> dWeight_momentum,dWeight_1,dWeight_2;
			mshadow::Tensor<xpu,1> dWeight;
			mshadow::Random<xpu> rnd_;
			int weight_size;
		protected:
			Wone(int size) :
					rnd_(0){
				Weight = mshadow::NewTensor<xpu>(mshadow::Shape1(size),0.01);
				dWeight = mshadow::NewTensor<xpu>(mshadow::Shape1(size),0.0);
				dWeight_2 = mshadow::NewTensor<xpu>(mshadow::Shape1(size),0.0);
				dWeight_1 = mshadow::NewTensor<xpu>(
						mshadow::Shape1(size),0.0);
				dWeight_momentum = mshadow::NewTensor<xpu>(
						mshadow::Shape1(size),0.0);

#ifdef DEBUG
				Weight = 0.01;
				//	LOG(INFO)<<typeid(Weight[0]).name();
#else
				rnd_.SampleGaussian(Weight);
				Weight *= 0.0005;
#endif

				ptr = Weight.dptr;
				dptr = dWeight.dptr;
			}
			Wone(Wone const&);
			void operator=(Wone const&);
			virtual ~Wone(){
			}
	}
	;
// data structures

	/*! \brief abstruct class for Node */
	template< typename xpu >
	class NodeFactory {
		public:
			NodeFactory(void){
				max_mem_ = std::numeric_limits<size_t>::max();
				warning_ = 1;
				total_mem_ = 0;
			}
			/* create new node */
			inline Node<xpu> CreateNode(void){
				return Node<xpu>(this);
			}
			/* set memory limits in terms of MB */
			inline void SetMemLimit(const char *size){
				if(!xpu::kDevCPU) {
					float n;
					if(sscanf(size,"%f",&n) == 1) {
						this->max_mem_ =
								static_cast<size_t>(n * (1L << 30) / 4);
						return;
					}
					warning_ = 1;
					mshadow::utils::Error("unknown memory limit string");
				}
			}
		private:
			friend class Node<xpu> ;
			/*! \brief request memory */
			inline void ReqMem(mshadow::Tensor<xpu,4> &data){
				size_t mem = data.shape.MSize();
				total_mem_ += mem;
				if(total_mem_ > max_mem_ && warning_ != 0) {
					printf(
							"warning: hit total memory limit, start swap mode\n");
					warning_ = 0;
				}
				while(total_mem_ > max_mem_) {
					CHECK(!free_list_.empty())
							<< "can not meet memory requirement";
					Node<xpu> *n = free_list_.front();
					free_list_.pop();
					CHECK(n->data.dptr != NULL) << "BUG";
					if(!n->pinned_) {
						if(n->backup_.dptr == NULL) {
							n->backup_.shape = n->data.shape;
							mshadow::AllocSpace(n->backup_);
						}
						mshadow::Copy(n->backup_,n->data);
						mshadow::FreeSpace(n->data);
						n->data.dptr = NULL;
					}
					n->inqueue_ = false;
				}
				mshadow::AllocSpace(data);
				total_mem_ += data.shape.MSize() - mem;
			}
			/*! \brief register the node as unpinned */
			inline void RegUnpin(Node<xpu> *n){
				if(n->inqueue_ == true) return;
				n->inqueue_ = true;
				free_list_.push(n);
			}
		private:
			/*! \brief whether do warning when memory swap occurs */
			int warning_;
			/*! \brief maximum memory allowed in total for nodes */
			size_t max_mem_;
			/*! \brief total amount of memory */
			size_t total_mem_;
			std::queue<Node<xpu>*> free_list_;
	};
// class NodeFactory

	template< typename xpu >
	struct Node {
		public:
			/*! \brief content of the node */
			mshadow::Tensor<xpu,4> data;
		public:
			/*! \brief free space of node */
			inline void FreeSpace(void){
				if(backup_.dptr != NULL) mshadow::FreeSpace(backup_);
				if(data.dptr != NULL) mshadow::FreeSpace(data);
			}
			/*! \brief matrix view of the node */
			inline mshadow::Tensor<xpu,2> mat(int t = 0){
				return data[0][t];
			}
			/*! \brief whether it holds a matrix data */
			inline bool is_mat(void) const{
				return data.shape[2] == 1 && data.shape[3] == 1;
			}
			/** before backprop reset in node */
			inline void resetNode(){
				data = 0.0;
			}
			inline void spyT(int t){
				for(unsigned int i = 0 ; i < data.shape[1] ; i++) {
					for(unsigned int j = 0 ; j < data.shape[0] ; j++) {
						LOG(INFO) << "T " << t << "sampele " << i << " "
								<< data[0][t][i][j];
					}
				}
			}
			inline void spy(){
				for(unsigned int i = 0 ; i < 2 ; i++) {
					for(unsigned int j = 0 ; j < data.shape[0] ; j++) {
						printf("sample%d %f ",i,data[0][0][i][j]);
					}
					printf("\n");
				}
				printf("\n");
			}
			/*! \brief pin the data into xpu memory,  will ensure it is there */
			inline void Pin(void){
				if(data.dptr != NULL) return;
				pinned_ = true;
				parent_->ReqMem(data);
				if(backup_.dptr != NULL) {
					mshadow::Copy(data,backup_);
				}
			}
			/*! \brief unpin the data, data can be deallocated */
			inline void Unpin(void){
				if(!pinned_) return;
				pinned_ = false;
				parent_->RegUnpin(this);
			}
		public:
			/* public constructor, use with caution */
			Node(void){
				data.dptr = NULL;
				backup_.dptr = NULL;
			}
		private:
			/*! \brief allow factory to see node */
			friend class NodeFactory<xpu> ;
			/*! \brief constructor */
			Node(NodeFactory<xpu>* parent) :
					parent_(parent){
				pinned_ = false;
				inqueue_ = false;
				data.dptr = NULL;
				backup_.dptr = NULL;
			}
			/*! \brief whether data is pinned */
			bool pinned_;
			/*! \brief whether data is in queue */
			bool inqueue_;
			/*! \brief pointer to parent */
			NodeFactory<xpu> *parent_;
			/*! \brief backup content of the node */
			mshadow::Tensor<mshadow::cpu,4> backup_;
	};
// struct Node

	template< typename xpu >
	ILayer<xpu>* CreateLayer(const LayerProto &l,mshadow::Random<xpu> &rnd,
			Node<xpu> &in,Node<xpu> &out);

}
;

#include "layer-inl.hpp"
#endif
