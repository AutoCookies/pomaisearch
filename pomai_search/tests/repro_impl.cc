#include "pomai_search/types.h"
#include "core/serialize/snapshot.h"
#include "src/search_engine_impl.h"
#include <iostream>

int main() {
    pomai_search::Shard::CandidateResult r;
    r.meta["foo"] = "bar";
    return 0;
}
