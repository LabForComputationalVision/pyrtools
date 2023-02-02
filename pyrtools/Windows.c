/**
 * This contains hacks to get the installation on Windows working.
 * This fixes error LNK2001: unresolved external symbol PyInit__cv_algorithms
 */
void PyInit_wrapConv(void) { }