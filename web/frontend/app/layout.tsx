import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'RAV25 · Occupancy Replay',
  description: 'Duyệt prediction OccFusion đã lưu trong hệ tọa độ LiDAR. Local, offline, không chạy lại mô hình.',
  icons: { icon: '/favicon.svg' },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
