
// This is required to allow the use of the python setuptools Extension
// class on Windows. Alternatives would be building ourselves or using
// a tool such as Meson which doesn't assume you're using the Python
// C API directly.
void* PyInit_wrapConv(void) {
    return 0;
}
