import { AppSidebar } from "@/components/app-sidebar";

import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";

export default function Page() {
  return (
    <div className="flex flex-1 flex-col gap-8 p-16 md:p-24">

  <div className="grid gap-8 md:grid-cols-2">
    <div className="wrapper flex flex-col justify-center">
      <h1 className="text-3xl font-semibold text-primary">
        Welcome to
      </h1>
      <h1 className="text-5xl font-bold text-accent">
        <span className="text-highlight">NaviGoose</span>
      </h1>
    </div>
    <div className="wrapper flex flex-col justify-center">
      <p className="text-lg text-secondary pt-4 md:pt-0">
        Navigate your way around mess-free with <span className="font-semibold text-accent">NaviGoose</span>
      </p>
    </div>
  </div>

  <div className="flex-1 mt-10 md:mt-16">
    {/* cam */}
  </div>

  <div className="min-h-[80vh] flex-1 rounded-xl bg-muted/50 shadow-lg md:min-h-min transition-all duration-500" />
</div>

  );
}
