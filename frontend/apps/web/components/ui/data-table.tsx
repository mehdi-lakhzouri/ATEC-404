"use client";

import { useMemo, useState, type ReactNode } from "react";
import {
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type PaginationState,
  type SortingState,
  type VisibilityState,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronFirst, ChevronLast, ChevronLeft, ChevronRight, Columns3, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

type DataTableProps<T> = {
  columns: ColumnDef<T>[];
  data: T[];
  getRowId: (row: T) => string;
  label: string;
  searchPlaceholder?: string;
  searchText?: (row: T) => string;
  searchQuery?: string;
  onSearchQueryChange?: (value: string) => void;
  toolbar?: ReactNode;
  emptyMessage?: string;
  onRowClick?: (row: T) => void;
  selectedRowId?: string | null;
  initialSorting?: SortingState;
  pageSize?: number;
};

/** shadcn Table primitives + TanStack sorting/visibility/pagination. Data remains owned by each API consumer. */
export function DataTable<T>({
  columns, data, getRowId, label, searchPlaceholder = "Search records…", searchText, searchQuery, onSearchQueryChange,
  toolbar, emptyMessage = "No matching records.", onRowClick, selectedRowId,
  initialSorting = [], pageSize = 10,
}: DataTableProps<T>) {
  const [internalQuery, setInternalQuery] = useState("");
  const query = searchQuery ?? internalQuery;
  const [sorting, setSorting] = useState<SortingState>(initialSorting);
  const [visibility, setVisibility] = useState<VisibilityState>({});
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize });
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return data;
    return data.filter((row) => {
      const text = searchText ? searchText(row) : Object.values(row as Record<string, unknown>)
        .filter((value) => typeof value === "string" || typeof value === "number").join(" ");
      return text.toLowerCase().includes(needle);
    });
  }, [data, query, searchText]);
  // Clamp on shrinking live data without jumping back to page one on every incoming packet.
  const boundedPagination = {
    ...pagination,
    pageIndex: Math.min(pagination.pageIndex, Math.max(0, Math.ceil(filtered.length / pagination.pageSize) - 1)),
  };
  const table = useReactTable({
    data: filtered, columns, getRowId,
    state: { sorting, columnVisibility: visibility, pagination: boundedPagination },
    onSortingChange: (updater) => {
      setSorting(updater);
      setPagination((p) => ({ ...p, pageIndex: 0 }));
    },
    onColumnVisibilityChange: setVisibility,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    autoResetPageIndex: false,
  });
  const pageCount = Math.max(1, table.getPageCount());
  const start = filtered.length ? boundedPagination.pageIndex * pagination.pageSize + 1 : 0;
  const end = Math.min(filtered.length, start + pagination.pageSize - 1);

  return (
    <div className="data-table" data-testid="data-table">
      <div className="data-table-toolbar">
        <div className="data-table-search">
          <Search aria-hidden="true" />
          <Input type="search" aria-label={`Search ${label}`} placeholder={searchPlaceholder} value={query}
            onChange={(e) => { setInternalQuery(e.target.value); onSearchQueryChange?.(e.target.value); setPagination((p) => ({ ...p, pageIndex: 0 })); }} />
        </div>
        {toolbar}
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" size="sm" />}>
            <Columns3 /> Columns
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {table.getAllLeafColumns().filter((column) => column.getCanHide()).map((column) => (
              <DropdownMenuCheckboxItem key={column.id} checked={column.getIsVisible()}
                disabled={column.getIsVisible() && table.getVisibleLeafColumns().length === 1}
                onCheckedChange={(checked) => column.toggleVisibility(checked)}>
                {typeof column.columnDef.header === "string" ? column.columnDef.header : column.id}
              </DropdownMenuCheckboxItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <Table aria-label={label}>
        <TableHeader>
          {table.getHeaderGroups().map((group) => (
            <TableRow key={group.id}>
              {group.headers.map((header) => (
                <TableHead key={header.id} aria-sort={header.column.getIsSorted() === "asc" ? "ascending" : header.column.getIsSorted() === "desc" ? "descending" : undefined}>
                  {header.isPlaceholder ? null : header.column.getCanSort() ? (
                    <Button variant="ghost" size="sm" className="data-table-sort" onClick={header.column.getToggleSortingHandler()}>
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" ? <ArrowUp /> : header.column.getIsSorted() === "desc" ? <ArrowDown /> : <ArrowUpDown />}
                    </Button>
                  ) : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows.length ? table.getRowModel().rows.map((row) => (
            <TableRow key={row.id} data-state={row.id === selectedRowId ? "selected" : undefined}
              className={cn(onRowClick && "data-table-interactive-row")}
              tabIndex={onRowClick ? 0 : undefined}
              onClick={onRowClick ? (e) => {
                if (!(e.target as HTMLElement).closest("button, a, input, [role=checkbox]")) onRowClick(row.original);
              } : undefined}
              onKeyDown={onRowClick ? (e) => {
                if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onRowClick(row.original); }
              } : undefined}>
              {row.getVisibleCells().map((cell) => <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>)}
            </TableRow>
          )) : (
            <TableRow><TableCell colSpan={table.getVisibleLeafColumns().length} className="data-table-empty">{emptyMessage}</TableCell></TableRow>
          )}
        </TableBody>
      </Table>
      <div className="data-table-pagination">
        <span className="data-table-count">{start}–{end} of {filtered.length} {query && `(${data.length} loaded)`}</span>
        <div className="data-table-page-size">
          <span>Rows per page</span>
          <Select value={String(pagination.pageSize)} onValueChange={(value) => value && table.setPageSize(Number(value))}>
            <SelectTrigger size="sm" aria-label={`Rows per page for ${label}`}><SelectValue /></SelectTrigger>
            <SelectContent>{[5, 10, 20, 50].map((size) => <SelectItem key={size} value={String(size)}>{size}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <span className="data-table-page">Page {boundedPagination.pageIndex + 1} of {pageCount}</span>
        <div className="data-table-page-buttons">
          <Button variant="outline" size="icon-sm" aria-label={`First page of ${label}`} disabled={!table.getCanPreviousPage()} onClick={() => table.firstPage()}><ChevronFirst /></Button>
          <Button variant="outline" size="icon-sm" aria-label={`Previous page of ${label}`} disabled={!table.getCanPreviousPage()} onClick={() => table.previousPage()}><ChevronLeft /></Button>
          <Button variant="outline" size="icon-sm" aria-label={`Next page of ${label}`} disabled={!table.getCanNextPage()} onClick={() => table.nextPage()}><ChevronRight /></Button>
          <Button variant="outline" size="icon-sm" aria-label={`Last page of ${label}`} disabled={!table.getCanNextPage()} onClick={() => table.lastPage()}><ChevronLast /></Button>
        </div>
      </div>
    </div>
  );
}
