#ifndef DATA_H
#define DATA_H
#pragma once

#include <vector>
#include <string>
#include <glog/logging.h>
#include "mshadow/tensor.h"
#include "../../layernet.pb.h"
#include "../compile_condition.h"

namespace layernet {
	/*!
	 * \brief iterator type
	 * \tparam DType data type
	 */
	template< typename DType >
	class IIterator {
		public:
			/*!
			 * \brief set the parameter
			 * \param name name of parameter
			 * \param val  value of parameter
			 */
			//  virtual void SetParam( const char *name, const char *val ) = 0;
			/*! \brief initalize the iterator so that we can use the iterator */
			virtual void Init(void) = 0;
			/*! \brief set before first of the item */
			virtual void BeforeFirst(void) = 0;
			/*! \brief move to next item */
			virtual bool Next(void) = 0;
			/*! \brief get current data */
			virtual const DType &Value(void) const = 0;
		public:
			/*! \brief constructor */
			virtual ~IIterator(void){
			}
	};

	/*! \brief a single data instance */
	struct DataInst {
			/*! \brief unique id for instance */
			unsigned index;
			/*! \brief content of data */
			mshadow::Tensor<mshadow::cpu,3> data,label;
	};

	/*! \brief a standard batch of data commonly used by iterator */
	struct DataBatch {
			/*! \brief label information */
			/*! \brief unique id for instance, can be NULL, sometimes is useful */
			unsigned* inst_index;
			/*! \brief number of instance */
			mshadow::index_t batch_size;
			/*! \brief content of data */
			mshadow::Tensor<mshadow::cpu,4> data,labels;
			mshadow::Tensor<mshadow::cpu,1> lengthlist;
			/*! \brief constructor */
			DataBatch(void){
				inst_index = NULL;
				batch_size = 0;
			}
			/*! \brief auxiliary to allocate space, if needed */
			inline void AllocSpace(mshadow::Shape<4> shape,
					mshadow::index_t batch_size,bool pad = false){
				data = mshadow::NewTensor<mshadow::cpu>(shape,0.0,pad);
				labels = mshadow::NewTensor<mshadow::cpu>(shape,0.0,pad);
				//		mshadow::Shape4(shape[3],shape[2],shape[1],1),0.0,pad);
				lengthlist = mshadow::NewTensor<mshadow::cpu>(
						mshadow::Shape1(shape[1]),0.0,pad);
				inst_index = new unsigned[batch_size];
				this->batch_size = batch_size;
			}
			/*! \brief auxiliary function to free space, if needed*/
			inline void FreeSpace(void){
				if(inst_index != NULL) {
					delete[] inst_index;
					mshadow::FreeSpace(data);
					mshadow::FreeSpace(labels);
					mshadow::FreeSpace(lengthlist);
				}
			}
			/*! \brief copy content from existing data */
			inline void CopyFrom(const DataBatch &src){
				CHECK(batch_size == src.batch_size);
				memcpy(inst_index,src.inst_index,batch_size * sizeof(unsigned));
				CHECK(data.shape == src.data.shape);
				mshadow::Copy(data,src.data);
				if(src.labels.dptr != NULL&&labels.shape == src.labels.shape)
					mshadow::Copy(labels,src.labels);

				if(src.lengthlist.dptr != NULL &&lengthlist.shape == src.lengthlist.shape)
					mshadow::Copy(lengthlist,src.lengthlist);

			}
	};
}
;

namespace layernet {
	/*!
	 * \brief create iterator from configure settings
	 * \param cfg configure settings key=vale pair
	 */
	IIterator<DataBatch>* CreateIterator(const DataProto &datap);
}
;
#endif
