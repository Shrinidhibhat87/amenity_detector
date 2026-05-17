export default function BrowseLoading() {
  return (
    <main className="flex-1 px-6 py-12 max-w-7xl mx-auto w-full">
      <div className="mb-8 space-y-2">
        <div className="h-3 w-16 rounded bg-surface-alt animate-pulse" />
        <div className="h-8 w-48 rounded bg-surface-alt animate-pulse" />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="rounded-2xl border border-border overflow-hidden">
            <div className="aspect-[4/3] bg-surface-alt animate-pulse" />
            <div className="p-4 space-y-2">
              <div className="h-4 w-3/4 rounded bg-surface-alt animate-pulse" />
              <div className="h-3 w-1/2 rounded bg-surface-alt animate-pulse" />
              <div className="h-3 w-1/3 rounded bg-surface-alt animate-pulse" />
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
