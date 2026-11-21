"""Windows GPU-engine and dedicated-memory counters by live process ID."""
import ctypes as c
import os
import re

class Value(c.Structure):
    _fields_=[("status",c.c_uint32),("number",c.c_double)]

class Item(c.Structure):
    _fields_=[("name",c.c_wchar_p),("value",Value)]


class ProcessGpu:
    def __init__(self):
        self.query=c.c_void_p()
        self.handles={}
        self.error=None
        if os.name!="nt":
            self.error="Windows GPU process counters unavailable"
            return
        try:
            self.dll=c.WinDLL("pdh.dll")
            status=self.dll.PdhOpenQueryW(None,0,c.byref(self.query))
            if status:
                raise OSError(f"PdhOpenQueryW: {status}")
            for key,path in (("engine",r"\GPU Engine(*)\Utilization Percentage"),
                             ("memory",r"\GPU Process Memory(*)\Dedicated Usage")):
                handle=c.c_void_p()
                if self.dll.PdhAddEnglishCounterW(self.query,c.c_wchar_p(path),0,c.byref(handle))==0:
                    self.handles[key]=handle
            if len(self.handles)!=2:
                self.error="One or more GPU process counters unavailable"
            self.dll.PdhCollectQueryData(self.query)
        except (AttributeError,OSError) as exc:
            self.error=str(exc)

    def values(self,handle):
        size=c.c_uint32()
        count=c.c_uint32()
        fn=self.dll.PdhGetFormattedCounterArrayW
        flags=0x200 # PDH_FMT_DOUBLE
        for _ in range(3):
            data=c.create_string_buffer(size.value) if size.value else None
            status=fn(handle,flags,c.byref(size),c.byref(count),data)
            if status & 0xffffffff==0x800007d2: # PDH_MORE_DATA
                continue
            if status or not data:
                return []
            items=c.cast(data,c.POINTER(Item))
            return [(items[i].name,items[i].value.number) for i in range(count.value)
                    if items[i].value.status in (0,1) and items[i].name]
        return []

    def sample(self,read_instances=True):
        processes={}
        if not self.handles or self.dll.PdhCollectQueryData(self.query):
            return processes
        if not read_instances:
            return processes
        for key,handle in self.handles.items():
            for name,value in self.values(handle):
                match=re.fullmatch(r"pid_(\d+)_(.+)",name,re.I)
                if not match:
                    continue
                row=processes.setdefault(int(match[1]),{})
                row.setdefault(key,{})[match[2].lower()]=max(0,value)
        return processes

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)

def attributed(processes,pids):
    memory=[]
    engines={}
    for pid in pids:
        row=processes.get(pid,{})
        memory.extend(row.get("memory",{}).values())
        for engine,pct in row.get("engine",{}).items():
            engines[engine]=engines.get(engine,0)+pct
    return (min(100.,max(engines.values())) if engines else None,
            int(sum(memory)) if memory else None)
