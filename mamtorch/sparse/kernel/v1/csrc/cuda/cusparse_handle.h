#ifndef CUSPARSE_HANDLE
#define CUSPARSE_HANDLE

#include <cusparse.h>

class CUSPARSEHandle
{
    public:
        CUSPARSEHandle() {
            cusparseCreate(&_handle);
        }

        ~CUSPARSEHandle() {
            cusparseDestroy(_handle);
        }

        cusparseHandle_t getHandle() const {
            return _handle;
        }

    private:
        cusparseHandle_t _handle;
};

#endif // CUSPARSE_HANDLE