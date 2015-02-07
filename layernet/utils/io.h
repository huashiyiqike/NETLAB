#ifndef CXXNET_IO_UTILS_H
#define CXXNET_IO_UTILS_H
/*!
 * \file cxxnet_io_utils.h
 * \brief io extensions
 * \author Bing Xu
 */
#include <zlib.h>
#include <fstream>
#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include "utils.h"
#include "mshadow/tensor.h"
#include <Python.h>

namespace layernet {
	class Python_helper {
		public:
			Python_helper(){
				Py_Initialize();
			}
			virtual ~Python_helper(void){
				//	if(img_.dptr != NULL) delete[] img_.dptr;
				Py_Finalize();
			}
			int load(mshadow::Tensor<mshadow::cpu,4> & data,long fileindex,
					int start,int end,std::string func){  // fileindex is the number of data.mat,
				//start, end is for batch, end - start should not exceed data.mat's length(100) and be divisable

				PyObject *sys = PyImport_ImportModule("sys");
				PyObject *path = PyObject_GetAttrString(sys,"path");
				PyList_Append(path,PyString_FromString("."));

				pName = PyString_FromString(func.c_str()); //Get the name of the module
				pModule = PyImport_Import(pName);     //Get the module
				Py_DECREF(pName);

				if(pModule != NULL) {
					pFunc = PyObject_GetAttrString(pModule,"func"); //Get the function by its name
					/* pFunc is a new reference */

					if(pFunc && PyCallable_Check(pFunc)) {

						//Set up a tuple that will contain the function arguments. In this case, the
						//function requires two tuples, so we set up a tuple of size 2.
						pArgTuple = PyTuple_New(1);
						std::vector<float> xvec(100);
						//Set the argument tuple to contain the two input tuples

						PyTuple_SetItem(pArgTuple,0,PyInt_FromLong(fileindex));
						//	PyTuple_SetItem(pArgTuple, 1, pYVec);

						//Call the python function
						if(PyCallable_Check(pFunc)) {
							pValue = PyObject_CallObject(pFunc,pArgTuple);

							// i is for batches
							for(int i = start ; i < end ; i++) {
								for(int ii = 0 ; ii < (int)data.shape[2] ; ii++) {
									for(int iii = 0 ; iii < (int)data.shape[0] ; iii++) {
										double tmp = PyFloat_AS_DOUBLE(
												PyList_GetItem(pValue,
														i * data.shape[2] * data.shape[0] + ii * data.shape[0]
																+ iii));
										data[0][ii][i - start][iii] = tmp;
									}
								}
							}
						}

						Py_DECREF(pArgTuple);
						//	Py_DECREF(pXVec);
						//	Py_DECREF(pYVec);

						if(pValue != NULL) {
							//	printf("Result of call: %ld\n",PyInt_AsLong(pValue));
							Py_DECREF(pValue);
						}

						//Some error catching
						else {
							Py_DECREF(pFunc);
							Py_DECREF(pModule);
							PyErr_Print();
							fprintf(stderr,"Call failed\n");
							return 1;
						}
					}
					else {
						if(PyErr_Occurred()) PyErr_Print();
						fprintf(stderr,"Cannot find function \n");
					}
					Py_XDECREF(pFunc);
					Py_DECREF(pModule);
				}
				else {
					PyErr_Print();
					fprintf(stderr,"Failed to load \n");
					return 1;
				}

				return 1;
			}
			int save(mshadow::Tensor<mshadow::cpu,4> & data,long T=100,long res=30){

				LOG(INFO)<< "Hello from runPython()"  ;

				PyObject *pName,*pModule,*pFunc;
				PyObject *pArgTuple,*pyVec,*pValue = NULL;

				// Set the path to include the current directory in case the module is located there. Found from
				// http://stackoverflow.com/questions/7624529/python-c-api-doesnt-load-module
				// and http://stackoverflow.com/questions/7283964/embedding-python-into-c-importing-modules
				PyObject *sys = PyImport_ImportModule("sys");
				PyObject *path = PyObject_GetAttrString(sys,"path");
				PyList_Append(path,PyString_FromString("."));

				pName = PyString_FromString("drawball"); //Get the name of the module
				pModule = PyImport_Import(pName);     //Get the module

				Py_DECREF(pName);

				if(pModule != NULL) {
					pFunc = PyObject_GetAttrString(pModule,"draw"); //Get the function by its name
					/* pFunc is a new reference */

					if(pFunc && PyCallable_Check(pFunc)) {

						//Set up a tuple that will contain the function arguments. In this case, the
						//function requires two tuples, so we set up a tuple of size 2.
						pArgTuple = PyTuple_New(3);
						std::vector<float> xvec(100);
						//Set the argument tuple to contain the two input tuples

						//Call the python function
						if(PyCallable_Check(pFunc)) {
							pyVec = PyTuple_New(data.shape[0] * data.shape[2]);
							int count = 0;
							for(int t = 0 ; t < T ; t++) {
								for(int i = 0 ; i < res ; i++) {
									for(int j = 0 ; j < res ; j++) {
										pValue = PyFloat_FromDouble(
												data[0][t][0][i * res + j]);
										if(!pValue) {
											Py_DECREF(pyVec);
											Py_DECREF(pModule);
											fprintf(stderr,
													"Cannot convert array value\n");
											return 1;
										}
										PyTuple_SetItem(pyVec,count++,pValue); //
									}
								}
							}
							PyTuple_SetItem(pArgTuple,0,pyVec);
							PyTuple_SetItem(pArgTuple,1,PyInt_FromLong(T));
							PyTuple_SetItem(pArgTuple,2,PyInt_FromLong(res));
							pValue = PyObject_CallObject(pFunc,pArgTuple);
							Py_DECREF(pArgTuple);
							//	Py_DECREF(pXVec);
							Py_DECREF(pyVec);
							if(pValue != NULL) {
								Py_DECREF(pValue);
							}
							//Some error catching
						}
						else {
							Py_DECREF(pFunc);
							Py_DECREF(pModule);
							PyErr_Print();
							fprintf(stderr,"Call failed\n");
							return 1;
						}
					}

					else {
						if(PyErr_Occurred()) PyErr_Print();
						fprintf(stderr,"Cannot find function \n");
					}
					Py_XDECREF(pFunc);
					Py_DECREF(pModule);
				}
				else {
					PyErr_Print();
					fprintf(stderr,"Failed to load \n");
					return 1;
				}

				return 1;
			}

		private:
			PyObject *pName,*pModule,*pFunc;
			PyObject *pArgTuple,*pValue;
	};

	inline bool ReadProtoFromTextFile(const char* filename,
			google::protobuf::Message* proto){
		using google::protobuf::io::FileInputStream;
		int fd = open(filename,O_RDONLY);
		CHECK_NE(fd,-1) << "File not found: " << filename;
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input,proto);
		delete input;
		close(fd);
		return success;
	}
	namespace utils {
		typedef mshadow::utils::IStream IStream;
		template< class T > static std::string str(const T& t){
			std::stringstream ss;
			ss << t;
			return ss.str();
			//	return lexical_cast<string>(t);
		}

		/*!
		 * \brief interface of stream that containes seek option,
		 *   mshadow does not rely on this interface
		 *   this is not always supported(e.g. in socket)
		 */
		class ISeekStream : public IStream {
			public:
				/*!
				 * \brief seek to a position, warning:
				 * \param pos relative position to start of stream
				 */
				virtual void Seek(size_t pos){
					utils::Error("Seek is not implemented");
				}
			public:
				inline int ReadInt(void){
					unsigned char buf[4];
					utils::Assert(Read(buf,sizeof(buf)) == sizeof(buf),
							"Failed to read an int\n");
					return int(
							buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
				}
				inline unsigned char ReadByte(void){
					unsigned char i;
					utils::Assert(Read(&i,sizeof(i)) == sizeof(i),
							"Failed to read an byte");
					return i;
				}
		};

		struct GzFile : public ISeekStream {
			public:
				GzFile(const char *path,const char *mode){
					fp_ = gzopen(path,mode);
					if(fp_ == NULL) {
						fprintf(stderr,"cannot open %s",path);
					}
					Assert(fp_ != NULL,"Failed to open file\n");
				}
				virtual ~GzFile(void){
					this->Close();
				}
				virtual void Close(void){
					if(fp_ != NULL) {
						gzclose(fp_);
						fp_ = NULL;
					}
				}
				virtual size_t Read(void *ptr,size_t size){
					return gzread(fp_,ptr,size);
				}
				virtual void Write(const void *ptr,size_t size){
					gzwrite(fp_,ptr,size);
				}
				virtual void Seek(size_t pos){
					gzseek(fp_,pos,SEEK_SET);
				}
			private:
				gzFile fp_;
		};

		/*! \brief implementation of file i/o stream */
		class StdFile : public ISeekStream {
			public:
				/*! \brief constructor */
				StdFile(const char *fname,const char *mode){
					Open(fname,mode);
				}
				StdFile(){
				}
				virtual void Open(const char *fname,const char *mode){
					fp_ = utils::FopenCheck(fname,mode);
					fseek(fp_,0L,SEEK_END);
					sz_ = ftell(fp_);
					fseek(fp_,0L,SEEK_SET);
				}
				virtual ~StdFile(void){
					this->Close();
				}
				virtual size_t Read(void *ptr,size_t size){
					return fread(ptr,size,1,fp_);
				}
				virtual void Write(const void *ptr,size_t size){
					fwrite(ptr,size,1,fp_);
				}
				virtual void Seek(size_t pos){
					fseek(fp_,pos,SEEK_SET);
				}
				inline void Close(void){
					if(fp_ != NULL) {
						fclose(fp_);
						fp_ = NULL;
					}
				}
				inline size_t Size(){
					return sz_;
				}
			private:
				FILE *fp_;
				size_t sz_;
		};
	}
	;
}
;

namespace layernet {
	namespace utils {
		/*! \brief Basic page class */
		class BinaryPage {
			public:
				/*! \brief page size 64 MB */
				static const size_t kPageSize = 64 << 18;
			public:
				/*! \brief memory data object */
				struct Obj {
						/*! \brief pointer to the data*/
						void *dptr;
						/*! \brief size */
						size_t sz;
						Obj(void * dptr,size_t sz) :
								dptr(dptr), sz(sz){
						}
				};
			public:
				/*! \brief constructor of page */
				BinaryPage(void){
					data_ = new int[kPageSize];
					utils::Assert(data_ != NULL);
					this->Clear();
				}
				;
				~BinaryPage(){
					if(data_) delete[] data_;
				}
				/*!
				 * \brief load one page form instream
				 * \return true if loading is successful
				 */
				inline bool Load(utils::IStream &fi){
					return fi.Read(&data_[0],sizeof(int) * kPageSize) != 0;
				}
				/*! \brief save one page into outstream */
				inline void Save(utils::IStream &fo){
					fo.Write(&data_[0],sizeof(int) * kPageSize);
				}
				/*! \return number of elements */
				inline int Size(void){
					return data_[0];
				}
				/*! \brief Push one binary object into page
				 *  \param fname file name of obj need to be pushed into
				 *  \return false or true to push into
				 */
				inline bool Push(const Obj &dat){
					if(this->FreeBytes() < dat.sz + sizeof(int)) return false;
					data_[Size() + 2] = data_[Size() + 1] + dat.sz;
					memcpy(this->offset(data_[Size() + 2]),dat.dptr,dat.sz);
					++data_[0];
					return true;
				}
				/*! \brief Clear the page */
				inline void Clear(void){
					memset(&data_[0],0,sizeof(int) * kPageSize);
				}
				/*!
				 * \brief Get one binary object from page
				 *  \param r r th obj in the page
				 */
				inline Obj operator[](int r){
					utils::Assert(r < Size(),"index excceed bound");
					return Obj(this->offset(data_[r + 2]),
							data_[r + 2] - data_[r + 1]);
				}
			private:
				/*! \return number of elements */
				inline size_t FreeBytes(void){
					return (kPageSize - (Size() + 2)) * sizeof(int)
							- data_[Size() + 1];
				}
				inline void* offset(int pos){
					return (char*) (&data_[0]) + (kPageSize * sizeof(int) - pos);
				}
			private:
				//int data_[ kPageSize ];
				int *data_;
		};
	// class BinaryPage
	}
	;
}
;
// namespace cxxnet
#endif
