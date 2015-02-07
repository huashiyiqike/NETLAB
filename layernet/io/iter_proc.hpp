#ifndef ITER_PROC_INL
#define ITER_PROC_INL
#pragma once
/*!
 * \file cxxnet_iter_proc-inl.hpp
 * \brief definition of preprocessing iterators that takes an iterator and do some preprocessing
 * \author Tianqi Chen
 */
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
#include "data.h"
#include "../utils/global_random.h"
#include "../utils/thread_buffer.h"

namespace layernet {

    /*! \brief thread buffer iterator */
    class ThreadBufferIterator: public IIterator< DataBatch >{
    public :
        ThreadBufferIterator( IIterator<DataBatch> *base,DataProto datap ){
            silent_ = 0;
            itr.get_factory().base_ = base;
            itr.SetParam( "buffer_size", "2" );
        }
        virtual ~ThreadBufferIterator(){
            itr.Destroy();
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "silent") ) silent_ = atoi( val );
            itr.SetParam( name, val );
        }
        virtual void Init( void ){
            utils::Assert( itr.Init() ) ;
            if( silent_ == 0 ){
                printf( "ThreadBufferIterator: buffer_size=%d\n", itr.buf_size );
            }
        }
        virtual void BeforeFirst(){
            itr.BeforeFirst();
        }
        virtual bool Next(){
            if( itr.Next( out_ ) ){
                return true;
            }else{
                return false;
            }
        }
        virtual const DataBatch &Value() const{
            return out_;
        }
    private:
        struct Factory{
        public:
            IIterator< DataBatch > *base_;
        public:
            Factory( void ){
                base_ = NULL;
            }
            inline void SetParam( const char *name, const char *val ){
              //  base_->SetParam( name, val );
            }
            inline bool Init(){
                base_->Init();
                utils::Assert( base_->Next(), "ThreadBufferIterator: input can not be empty" );
                oshape_ = base_->Value().data.shape;
                batch_size_ = base_->Value().batch_size;
                base_->BeforeFirst();
                return true;
            }
            inline bool LoadNext( DataBatch &val ){
                if( base_->Next() ){
                    val.CopyFrom( base_->Value() );
                    return true;
                }else{
                    return false;
                }
            }
            inline DataBatch Create( void ){
                DataBatch a; a.AllocSpace( oshape_, batch_size_ );
                return a;
            }
            inline void FreeSpace( DataBatch &a ){
                a.FreeSpace();
            }
            inline void Destroy(){
                if( base_ != NULL ) delete base_;
            }
            inline void BeforeFirst(){
                base_->BeforeFirst();
            }
        private:
            mshadow::index_t batch_size_;
            mshadow::Shape<4> oshape_;
        };
    private:
        int silent_;
        DataBatch out_;
        utils::ThreadBuffer<DataBatch,Factory> itr;
    };
};
#endif
