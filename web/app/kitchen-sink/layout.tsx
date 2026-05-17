import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kitchen sink — Amenity Detector',
  robots: { index: false, follow: false },
};

export default function KitchenSinkLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
