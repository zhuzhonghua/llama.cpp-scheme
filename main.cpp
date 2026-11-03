// scmllama: minimal host that runs a dummyscheme script with llama.cpp bindings.
//

#include "scmllama.h"

#include <cstdio>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    fprintf(stderr, "usage: %s file.scm\n", argv[0]);
    return 2;
  }

  int rc = 0;
  {
    // Construction creates the VM and initializes the llama backend;
    // destruction tears them down in reverse order.
    scmllama::Runtime runtime;

    if (!runtime.vm())
    {
      fprintf(stderr, "failed to create VM\n");
      return 1;
    }

    TRY
    {
      runtime.vm()->loadfile(argv[1]);
    }
    CATCH(err)
    {
      fprintf(stderr, "error: %s\n", err ? err : "unknown");
      rc = 1;
    }
  }
  return rc;
}
