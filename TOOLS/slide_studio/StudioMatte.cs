using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public class StudioMatte {
    private static byte[] Pixels(Bitmap bmp) {
        var data = bmp.LockBits(new Rectangle(0,0,bmp.Width,bmp.Height),ImageLockMode.ReadOnly,PixelFormat.Format32bppArgb);
        try {
            byte[] bytes = new byte[bmp.Width*bmp.Height*4];
            for(int y=0;y<bmp.Height;y++) Marshal.Copy(IntPtr.Add(data.Scan0,y*data.Stride),bytes,y*bmp.Width*4,bmp.Width*4);
            return bytes;
        } finally {bmp.UnlockBits(data);}
    }
    // A black and white exposure isolate alpha independently of the authoring bounding box.
    public static int[] Save(string dark,string light,string dest) {
        using(var b=new Bitmap(dark)) using(var w=new Bitmap(light)){
            var bg=Pixels(b);var fg=Pixels(w);
            int left=b.Width,top=b.Height,right=-1,bottom=-1;
            for(int y=0;y<b.Height;y++)for(int x=0;x<b.Width;x++){
                int i=(y*b.Width+x)*4;
                int a=Math.Max(0,Math.Min(255,255-(fg[i]-bg[i]+fg[i+1]-bg[i+1]+fg[i+2]-bg[i+2]+1)/3));
                if(a<2){bg[i+3]=0;continue;}
                bg[i+3]=(byte)a;
                for(int c=0;c<3;c++)bg[i+c]=(byte)Math.Min(255,(bg[i+c]*255+a/2)/a);
                if(x<left)left=x;if(x>right)right=x;if(y<top)top=y;if(y>bottom)bottom=y;
            }
            if(right<left)return new int[0];
            var bounds=new Rectangle(left,top,right-left+1,bottom-top+1);
            using(var cropped=new Bitmap(bounds.Width,bounds.Height,PixelFormat.Format32bppArgb)){
                var data=cropped.LockBits(new Rectangle(0,0,bounds.Width,bounds.Height),ImageLockMode.WriteOnly,PixelFormat.Format32bppArgb);
                try{for(int y=0;y<bounds.Height;y++)Marshal.Copy(bg,((y+top)*b.Width+left)*4,IntPtr.Add(data.Scan0,y*data.Stride),bounds.Width*4);}
                finally{cropped.UnlockBits(data);}
                cropped.Save(dest,ImageFormat.Png);
            }
            return new[]{left,top,bounds.Width,bounds.Height};
        }
    }
}
